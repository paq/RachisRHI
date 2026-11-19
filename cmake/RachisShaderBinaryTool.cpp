#include "Internal/ShaderBinaryTool.h"
#include "Internal/ShaderInterfaceReflection.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern "C" char** environ;
#endif

namespace
{

    [[nodiscard]] std::filesystem::path pathFromUtf8(std::string_view text)
    {
        return std::filesystem::path(std::u8string(text.begin(), text.end()));
    }

    [[nodiscard]] std::string pathToUtf8(const std::filesystem::path& path)
    {
        const std::u8string text = path.u8string();
        return std::string(text.begin(), text.end());
    }

    void printUsage(std::ostream& stream)
    {
        stream << "Usage:\n"
               << "  RachisShaderBinaryTool --help\n"
               << "  RachisShaderBinaryTool embed-header --input <binary> --output <header> --symbol <identifier>\n"
               << "  RachisShaderBinaryTool pack-pnms --json <reflection.json> --metallib <shader.metallib> --output "
                  "<shader.pnms>\n"
               << "  RachisShaderBinaryTool cook-metallib --slangc <slangc> --source <shader> --entry <entry> --output "
                  "<shader.metallib> --json <reflection.json> --pnms <shader.pnms> [--stage <stage>] "
                  "[--metal-source <shader.metal>] [--depfile <shader.d>] [--no-warnings-as-errors] "
                  "[--define <NAME=VALUE> ...] [--include-dir <directory> ...]\n"
               << "  RachisShaderBinaryTool reflect-interface --name <identifier> --stage <stage> --entry <entry> "
                  "--output <header> --reflection <backend>=<reflection.json> ... [--layout <backend>=<file> ...] "
                  "[--spirv-reflection vulkan=<module.spv>] [--bytecode <d3d12|vulkan>=<file> ...] [--tag <tag>]\n";
    }

    [[nodiscard]] std::expected<std::vector<std::byte>, std::string> readBinaryFile(const std::string& rawPath)
    {
        if (rawPath.empty())
        {
            return std::unexpected("Input path must not be empty.");
        }
        const std::filesystem::path path = pathFromUtf8(rawPath);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
        {
            return std::unexpected("Input file does not exist: " + rawPath);
        }
        if (std::filesystem::is_directory(path, ec))
        {
            return std::unexpected("Input path is a directory: " + rawPath);
        }

        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream.is_open() || !stream.good())
        {
            return std::unexpected("Failed to open input file: " + rawPath);
        }

        const std::streampos endPosition = stream.tellg();
        if (endPosition < 0)
        {
            return std::unexpected("Failed to query size of input file: " + rawPath);
        }
        const auto rawSize = static_cast<std::uintmax_t>(endPosition);
        if (rawSize > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
            rawSize > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        {
            return std::unexpected("Input file exceeds addressable size: " + rawPath);
        }

        const auto size = static_cast<std::size_t>(rawSize);
        stream.seekg(0, std::ios::beg);
        if (!stream.good())
        {
            return std::unexpected("Failed to seek input file: " + rawPath);
        }

        std::vector<std::byte> bytes(size);
        if (size > 0)
        {
            stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
            if (!stream || static_cast<std::size_t>(stream.gcount()) != size)
            {
                return std::unexpected("Failed to read input file: " + rawPath);
            }
        }
        return bytes;
    }

    [[nodiscard]] std::expected<void, std::string> writeBinaryFile(const std::string& rawPath,
                                                                   std::span<const std::byte> data)
    {
        if (rawPath.empty())
        {
            return std::unexpected("Output path must not be empty.");
        }
        std::error_code ec;
        const std::filesystem::path output = std::filesystem::absolute(pathFromUtf8(rawPath), ec);
        if (ec)
        {
            return std::unexpected("Failed to resolve output path '" + rawPath + "': " + ec.message());
        }
        if (std::filesystem::is_directory(output, ec))
        {
            return std::unexpected("Output path is an existing directory: " + pathToUtf8(output));
        }

        const std::filesystem::path parent = output.parent_path();
        if (!parent.empty())
        {
            std::filesystem::create_directories(parent, ec);
            if (ec)
            {
                return std::unexpected("Failed to create output directory '" + pathToUtf8(parent) +
                                       "': " + ec.message());
            }
        }

        std::ofstream stream(output, std::ios::binary | std::ios::trunc);
        if (!stream.is_open() || !stream.good())
        {
            return std::unexpected("Failed to open output file: " + pathToUtf8(output));
        }
        if (!data.empty())
        {
            stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
        stream.flush();
        if (!stream.good())
        {
            return std::unexpected("Failed to write output file: " + pathToUtf8(output));
        }
        return {};
    }

    [[nodiscard]] int runEmbedHeader(std::span<const std::string_view> args)
    {
        if (args.size() == 1 && args[0] == "--help")
        {
            std::cout << "--input <binary> --output <header> --symbol <identifier>\n";
            return 0;
        }

        std::string inputPath;
        std::string outputPath;
        std::string symbol;
        bool hasInput = false;
        bool hasOutput = false;
        bool hasSymbol = false;

        for (std::size_t index = 0; index < args.size(); index += 2)
        {
            if (index + 1 == args.size())
            {
                std::cerr << "Expected --input, --output, and --symbol exactly once.\n";
                return 1;
            }
            const std::string_view flag = args[index];
            const std::string_view value = args[index + 1];
            if (flag == "--input" && !hasInput)
            {
                hasInput = true;
                inputPath.assign(value);
            }
            else if (flag == "--output" && !hasOutput)
            {
                hasOutput = true;
                outputPath.assign(value);
            }
            else if (flag == "--symbol" && !hasSymbol)
            {
                hasSymbol = true;
                symbol.assign(value);
            }
            else
            {
                std::cerr << "Expected --input, --output, and --symbol exactly once.\n";
                return 1;
            }
        }

        if (!hasInput || !hasOutput || !hasSymbol)
        {
            std::cerr << "--input, --output, and --symbol are required.\n";
            return 1;
        }

        const auto inputBytes = readBinaryFile(inputPath);
        if (!inputBytes.has_value())
        {
            std::cerr << inputBytes.error() << '\n';
            return 1;
        }

        const auto headerText = rachis::shader_tool::generateEmbeddedBinaryHeader(*inputBytes, symbol);
        if (!headerText.has_value())
        {
            std::cerr << headerText.error() << '\n';
            return 1;
        }

        const std::span<const std::byte> outputBytes(reinterpret_cast<const std::byte*>(headerText->data()),
                                                     headerText->size());
        const auto writeStatus = writeBinaryFile(outputPath, outputBytes);
        if (!writeStatus.has_value())
        {
            std::cerr << writeStatus.error() << '\n';
            return 1;
        }
        return 0;
    }

    [[nodiscard]] int runPackPnms(std::span<const std::string_view> args)
    {
        if (args.size() == 1 && args[0] == "--help")
        {
            std::cout << "--json <reflection.json> --metallib <shader.metallib> --output <shader.pnms>\n";
            return 0;
        }

        std::string jsonPath;
        std::string metallibPath;
        std::string outputPath;
        bool hasJson = false;
        bool hasMetallib = false;
        bool hasOutput = false;

        for (std::size_t index = 0; index < args.size(); index += 2)
        {
            if (index + 1 == args.size())
            {
                std::cerr << "Expected --json, --metallib, and --output exactly once.\n";
                return 1;
            }
            const std::string_view flag = args[index];
            const std::string_view value = args[index + 1];
            if (flag == "--json" && !hasJson)
            {
                hasJson = true;
                jsonPath.assign(value);
            }
            else if (flag == "--metallib" && !hasMetallib)
            {
                hasMetallib = true;
                metallibPath.assign(value);
            }
            else if (flag == "--output" && !hasOutput)
            {
                hasOutput = true;
                outputPath.assign(value);
            }
            else
            {
                std::cerr << "Expected --json, --metallib, and --output exactly once.\n";
                return 1;
            }
        }

        if (!hasJson || !hasMetallib || !hasOutput)
        {
            std::cerr << "--json, --metallib, and --output are required.\n";
            return 1;
        }

        const auto reflectionBytes = readBinaryFile(jsonPath);
        if (!reflectionBytes.has_value())
        {
            std::cerr << reflectionBytes.error() << '\n';
            return 1;
        }

        const auto metallibBytes = readBinaryFile(metallibPath);
        if (!metallibBytes.has_value())
        {
            std::cerr << metallibBytes.error() << '\n';
            return 1;
        }

        const auto packedBytes = rachis::shader_tool::packMetalShaderManifest(*reflectionBytes, *metallibBytes);
        if (!packedBytes.has_value())
        {
            std::cerr << packedBytes.error() << '\n';
            return 1;
        }

        const auto writeStatus = writeBinaryFile(outputPath, *packedBytes);
        if (!writeStatus.has_value())
        {
            std::cerr << writeStatus.error() << '\n';
            return 1;
        }
        return 0;
    }

    [[nodiscard]] std::string quoteCommandLineArgument(std::string_view arg)
    {
        if (!arg.empty() && arg.find_first_of(" \t\"") == std::string_view::npos)
        {
            return std::string(arg);
        }
        std::string quoted = "\"";
        std::size_t backslashes = 0;
        for (const char ch : arg)
        {
            if (ch == '\\')
            {
                ++backslashes;
            }
            else if (ch == '"')
            {
                quoted.append(backslashes * 2 + 1, '\\');
                quoted.push_back('"');
                backslashes = 0;
            }
            else
            {
                quoted.append(backslashes, '\\');
                quoted.push_back(ch);
                backslashes = 0;
            }
        }
        quoted.append(backslashes * 2, '\\');
        quoted.push_back('"');
        return quoted;
    }

    [[nodiscard]] std::string formatCommandLine(const std::string& executable,
                                                const std::vector<std::string>& arguments)
    {
        std::string commandLine = quoteCommandLineArgument(executable);
        for (const auto& arg : arguments)
        {
            commandLine.push_back(' ');
            commandLine += quoteCommandLineArgument(arg);
        }
        return commandLine;
    }

#ifdef _WIN32
    [[nodiscard]] std::wstring widenUtf8(std::string_view text)
    {
        if (text.empty() || text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return {};
        }
        const int textSize = static_cast<int>(text.size());
        const int wideSize = MultiByteToWideChar(CP_UTF8, 0, text.data(), textSize, nullptr, 0);
        if (wideSize <= 0)
        {
            return {};
        }
        std::wstring wide(static_cast<std::size_t>(wideSize), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, text.data(), textSize, wide.data(), wideSize);
        return wide;
    }

    [[nodiscard]] std::string narrowToUtf8(std::wstring_view text)
    {
        if (text.empty() || text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return {};
        }
        const int textSize = static_cast<int>(text.size());
        const int narrowSize = WideCharToMultiByte(CP_UTF8, 0, text.data(), textSize, nullptr, 0, nullptr, nullptr);
        if (narrowSize <= 0)
        {
            return {};
        }
        std::string narrow(static_cast<std::size_t>(narrowSize), '\0');
        WideCharToMultiByte(CP_UTF8, 0, text.data(), textSize, narrow.data(), narrowSize, nullptr, nullptr);
        return narrow;
    }

    [[nodiscard]] std::string describeWindowsError(DWORD error)
    {
        return std::system_category().message(static_cast<int>(error)) + " (GetLastError " + std::to_string(error) +
               ")";
    }

    [[nodiscard]] std::expected<int, std::string> runWindowsProcess(
        const std::string& executable,
        const std::vector<std::string>& arguments,
        const std::optional<std::string>& capturedOutputPath)
    {
        const std::string commandLine = formatCommandLine(executable, arguments);
        std::wstring wideCommandLine = widenUtf8(commandLine);
        STARTUPINFOW startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        HANDLE outputHandle = INVALID_HANDLE_VALUE;
        if (capturedOutputPath.has_value())
        {
            SECURITY_ATTRIBUTES inheritable{};
            inheritable.nLength = sizeof(inheritable);
            inheritable.bInheritHandle = TRUE;
            outputHandle = CreateFileW(widenUtf8(*capturedOutputPath).c_str(),
                                       GENERIC_WRITE,
                                       FILE_SHARE_READ,
                                       &inheritable,
                                       CREATE_ALWAYS,
                                       FILE_ATTRIBUTE_NORMAL,
                                       nullptr);
            if (outputHandle == INVALID_HANDLE_VALUE)
            {
                return std::unexpected("Failed to create process output file '" + *capturedOutputPath +
                                       "': " + describeWindowsError(GetLastError()));
            }
            startupInfo.dwFlags = STARTF_USESTDHANDLES;
            startupInfo.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            startupInfo.hStdOutput = outputHandle;
            startupInfo.hStdError = outputHandle;
        }

        PROCESS_INFORMATION processInfo{};
        const BOOL created = CreateProcessW(
            nullptr, wideCommandLine.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startupInfo, &processInfo);
        const DWORD createError = created ? ERROR_SUCCESS : GetLastError();
        if (outputHandle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(outputHandle);
        }
        if (!created)
        {
            return std::unexpected("Failed to start " + commandLine + ": " + describeWindowsError(createError));
        }
        CloseHandle(processInfo.hThread);

        DWORD exitCode = 0;
        const bool exited = WaitForSingleObject(processInfo.hProcess, INFINITE) == WAIT_OBJECT_0 &&
                            GetExitCodeProcess(processInfo.hProcess, &exitCode) != 0;
        const DWORD waitError = exited ? ERROR_SUCCESS : GetLastError();
        CloseHandle(processInfo.hProcess);
        if (!exited)
        {
            return std::unexpected("Failed to wait for " + commandLine + ": " + describeWindowsError(waitError));
        }
        return static_cast<int>(exitCode);
    }
#else
    [[nodiscard]] std::string describeErrno(int error)
    {
        return std::generic_category().message(error) + " (errno " + std::to_string(error) + ")";
    }

    [[nodiscard]] int waitForExitCode(pid_t pid)
    {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0)
        {
            if (errno != EINTR)
            {
                return 1;
            }
        }
        if (WIFEXITED(status))
        {
            return WEXITSTATUS(status);
        }
        return 1;
    }

    [[nodiscard]] std::expected<int, std::string> runPosixProcess(const std::string& executable,
                                                                  const std::vector<std::string>& arguments,
                                                                  const std::optional<std::string>& capturedOutputPath)
    {
        const std::string commandLine = formatCommandLine(executable, arguments);
        std::vector<std::string> storage;
        storage.reserve(arguments.size() + 1);
        storage.push_back(executable);
        for (const auto& arg : arguments)
        {
            storage.push_back(arg);
        }
        std::vector<char*> argv;
        argv.reserve(storage.size() + 1);
        for (auto& item : storage)
        {
            argv.push_back(item.data());
        }
        argv.push_back(nullptr);

        posix_spawn_file_actions_t actions{};
        posix_spawn_file_actions_t* actionsPtr = nullptr;
        if (capturedOutputPath.has_value())
        {
            if (const int initError = posix_spawn_file_actions_init(&actions); initError != 0)
            {
                return std::unexpected("Failed to prepare " + commandLine + ": " + describeErrno(initError));
            }
            actionsPtr = &actions;
            int actionError = posix_spawn_file_actions_addopen(
                &actions, STDOUT_FILENO, capturedOutputPath->c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (actionError == 0)
            {
                actionError = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
            }
            if (actionError != 0)
            {
                (void)posix_spawn_file_actions_destroy(&actions);
                return std::unexpected("Failed to prepare " + commandLine + ": " + describeErrno(actionError));
            }
        }

        pid_t pid = 0;
        const int spawnError = posix_spawnp(&pid, executable.c_str(), actionsPtr, nullptr, argv.data(), environ);
        if (actionsPtr != nullptr)
        {
            (void)posix_spawn_file_actions_destroy(&actions);
        }
        if (spawnError != 0)
        {
            return std::unexpected("Failed to start " + commandLine + ": " + describeErrno(spawnError));
        }
        return waitForExitCode(pid);
    }
#endif

    [[nodiscard]] std::expected<int, std::string> runProcess(const std::string& executable,
                                                             const std::vector<std::string>& arguments,
                                                             const std::optional<std::string>& capturedOutputPath)
    {
        if (executable.empty())
        {
            return std::unexpected("Executable path must not be empty.");
        }
#ifdef _WIN32
        return runWindowsProcess(executable, arguments, capturedOutputPath);
#else
        return runPosixProcess(executable, arguments, capturedOutputPath);
#endif
    }

    [[nodiscard]] int runCompilerStep(const std::string& executable, const std::vector<std::string>& arguments)
    {
        const auto exitCode = runProcess(executable, arguments, std::nullopt);
        if (!exitCode.has_value())
        {
            std::cerr << exitCode.error() << '\n';
            return 1;
        }
        return *exitCode;
    }

    struct ScopedTempFileCleanup final
    {
        std::vector<std::filesystem::path> paths{};

        ScopedTempFileCleanup() = default;
        ScopedTempFileCleanup(const ScopedTempFileCleanup&) = delete;
        ScopedTempFileCleanup& operator=(const ScopedTempFileCleanup&) = delete;
        ScopedTempFileCleanup(ScopedTempFileCleanup&&) = delete;
        ScopedTempFileCleanup& operator=(ScopedTempFileCleanup&&) = delete;

        ~ScopedTempFileCleanup()
        {
            std::error_code ec;
            for (const auto& path : paths)
            {
                std::filesystem::remove(path, ec);
            }
        }
    };

    struct CookMetallibArguments final
    {
        std::optional<std::string> slangcPath;
        std::optional<std::string> sourcePath;
        std::optional<std::string> entryPoint;
        std::optional<std::string> outputPath;
        std::optional<std::string> jsonPath;
        std::optional<std::string> pnmsPath;
        std::optional<std::string> stage;
        std::optional<std::string> metalSourcePath;
        std::optional<std::string> depfilePath;
        std::vector<std::string> defines{};
        std::vector<std::string> includeDirectories{};
        bool noWarningsAsErrors = false;
    };

    [[nodiscard]] bool assignOnce(std::optional<std::string>& destination, std::string_view value)
    {
        if (destination.has_value())
        {
            return false;
        }
        destination.emplace(value);
        return true;
    }

    [[nodiscard]] bool assignCookMetallibArgument(CookMetallibArguments& arguments,
                                                  std::string_view flag,
                                                  std::string_view value)
    {
        if (flag == "--slangc")
        {
            return assignOnce(arguments.slangcPath, value);
        }
        if (flag == "--source")
        {
            return assignOnce(arguments.sourcePath, value);
        }
        if (flag == "--entry")
        {
            return assignOnce(arguments.entryPoint, value);
        }
        if (flag == "--output")
        {
            return assignOnce(arguments.outputPath, value);
        }
        if (flag == "--json")
        {
            return assignOnce(arguments.jsonPath, value);
        }
        if (flag == "--pnms")
        {
            return assignOnce(arguments.pnmsPath, value);
        }
        if (flag == "--stage")
        {
            return assignOnce(arguments.stage, value);
        }
        if (flag == "--metal-source")
        {
            return assignOnce(arguments.metalSourcePath, value);
        }
        if (flag == "--depfile")
        {
            return assignOnce(arguments.depfilePath, value);
        }
        if (flag == "--define")
        {
            arguments.defines.emplace_back(value);
            return true;
        }
        if (flag == "--include-dir")
        {
            arguments.includeDirectories.emplace_back(value);
            return true;
        }
        return false;
    }

    [[nodiscard]] std::expected<CookMetallibArguments, std::string> collectCookMetallibArguments(
        std::span<const std::string_view> args)
    {
        CookMetallibArguments arguments;
        for (std::size_t index = 0; index < args.size(); ++index)
        {
            const std::string_view flag = args[index];
            if (flag == "--no-warnings-as-errors" && !arguments.noWarningsAsErrors)
            {
                arguments.noWarningsAsErrors = true;
                continue;
            }
            if (index + 1 >= args.size())
            {
                return std::unexpected("Missing value for argument: " + std::string(flag));
            }
            const std::string_view value = args[++index];
            if (!assignCookMetallibArgument(arguments, flag, value))
            {
                return std::unexpected("Unexpected or duplicate cook-metallib argument: " + std::string(flag));
            }
        }
        return arguments;
    }

    struct CookMetallibOptions final
    {
        std::string slangcPath;
        std::string sourcePath;
        std::string entryPoint;
        std::string outputPath;
        std::string jsonPath;
        std::string pnmsPath;
        std::optional<std::string> stage;
        std::optional<std::string> metalSourcePath;
        std::optional<std::string> depfilePath;
        std::vector<std::string> defines{};
        std::vector<std::string> includeDirectories{};
        bool noWarningsAsErrors = false;
    };

    [[nodiscard]] std::expected<CookMetallibOptions, std::string> parseCookMetallibOptions(
        std::span<const std::string_view> args)
    {
        auto arguments = collectCookMetallibArguments(args);
        if (!arguments.has_value())
        {
            return std::unexpected(std::move(arguments.error()));
        }
        if (!arguments->slangcPath.has_value() || !arguments->sourcePath.has_value() ||
            !arguments->entryPoint.has_value() || !arguments->outputPath.has_value() ||
            !arguments->jsonPath.has_value() || !arguments->pnmsPath.has_value())
        {
            return std::unexpected("--slangc, --source, --entry, --output, --json, and --pnms are required.");
        }
        return CookMetallibOptions{
            .slangcPath = std::move(*arguments->slangcPath),
            .sourcePath = std::move(*arguments->sourcePath),
            .entryPoint = std::move(*arguments->entryPoint),
            .outputPath = std::move(*arguments->outputPath),
            .jsonPath = std::move(*arguments->jsonPath),
            .pnmsPath = std::move(*arguments->pnmsPath),
            .stage = std::move(arguments->stage),
            .metalSourcePath = std::move(arguments->metalSourcePath),
            .depfilePath = std::move(arguments->depfilePath),
            .defines = std::move(arguments->defines),
            .includeDirectories = std::move(arguments->includeDirectories),
            .noWarningsAsErrors = arguments->noWarningsAsErrors,
        };
    }

    [[nodiscard]] std::expected<std::filesystem::path, std::string> prepareOutputPath(const std::string& outputPath)
    {
        std::error_code ec;
        std::filesystem::path outputFs = std::filesystem::absolute(pathFromUtf8(outputPath), ec);
        if (ec)
        {
            return std::unexpected("Failed to resolve output path '" + outputPath + "': " + ec.message());
        }
        if (const auto parent = outputFs.parent_path(); !parent.empty())
        {
            std::filesystem::create_directories(parent, ec);
            if (ec)
            {
                return std::unexpected("Failed to create output directory '" + pathToUtf8(parent) +
                                       "': " + ec.message());
            }
        }
        return outputFs;
    }

    struct CookTempFiles final
    {
        std::string spirvProbeJson;
        std::string spirvProbeLog;
        std::string metalProbeJson;
        std::string metalProbeLog;
        std::string metalProbeSource;
        std::string rawReflectionJson;
        std::string rebasedMetalSource;
    };

    [[nodiscard]] CookTempFiles makeCookTempFiles(const std::filesystem::path& outputFs)
    {
        const std::string output = pathToUtf8(outputFs);
        return CookTempFiles{
            .spirvProbeJson = output + ".spv-probe.tmp.json",
            .spirvProbeLog = output + ".spv-probe.tmp.log",
            .metalProbeJson = output + ".mtl0-probe.tmp.json",
            .metalProbeLog = output + ".mtl0-probe.tmp.log",
            .metalProbeSource = output + ".mtl0-probe.tmp.metal",
            .rawReflectionJson = output + ".raw.tmp.json",
            .rebasedMetalSource = output + ".rebased.tmp.metal",
        };
    }

    void scheduleCookTempFileCleanup(ScopedTempFileCleanup& cleanup, const CookTempFiles& tempFiles)
    {
        cleanup.paths = {
            pathFromUtf8(tempFiles.spirvProbeJson),
            pathFromUtf8(tempFiles.spirvProbeLog),
            pathFromUtf8(tempFiles.metalProbeJson),
            pathFromUtf8(tempFiles.metalProbeLog),
            pathFromUtf8(tempFiles.metalProbeSource),
            pathFromUtf8(tempFiles.rawReflectionJson),
            pathFromUtf8(tempFiles.rebasedMetalSource),
        };
    }

    void appendDefineArgs(std::vector<std::string>& args, const std::vector<std::string>& defines)
    {
        for (const auto& def : defines)
        {
            args.push_back("-D" + def);
        }
    }

    void appendIncludeArgs(std::vector<std::string>& args, const CookMetallibOptions& options)
    {
        for (const auto& directory : options.includeDirectories)
        {
            args.push_back("-I");
            args.push_back(directory);
        }
    }

    void appendEntryArgs(std::vector<std::string>& args, const CookMetallibOptions& options)
    {
        args.push_back("-entry");
        args.push_back(options.entryPoint);
        if (options.stage.has_value())
        {
            args.push_back("-stage");
            args.push_back(*options.stage);
        }
    }

    [[nodiscard]] std::vector<std::string> buildSpirvProbeArgs(const CookMetallibOptions& options,
                                                               const CookTempFiles& tempFiles)
    {
        std::vector<std::string> args{
            options.sourcePath,
            "-target",
            "spirv",
            "-no-codegen",
            "-profile",
            "sm_6_0",
            "-matrix-layout-column-major",
            "-fvk-use-entrypoint-name",
            "-Wno-39029",
            "-Wno-39001",
            "-Wno-40100",
        };
        appendDefineArgs(args, options.defines);
        appendIncludeArgs(args, options);
        appendEntryArgs(args, options);
        args.insert(args.end(), {"-reflection-json", tempFiles.spirvProbeJson});
        return args;
    }

    [[nodiscard]] std::vector<std::string> buildMetalProbeArgs(const CookMetallibOptions& options,
                                                               const CookTempFiles& tempFiles)
    {
        std::vector<std::string> args{
            options.sourcePath,
            "-target",
            "metal",
            "-o",
            tempFiles.metalProbeSource,
            "-profile",
            "sm_6_0",
            "-matrix-layout-column-major",
            "-fvk-use-entrypoint-name",
            "-fvk-b-shift",
            "0",
            "all",
            "-fvk-t-shift",
            "0",
            "all",
            "-fvk-s-shift",
            "0",
            "all",
            "-fvk-u-shift",
            "0",
            "all",
            "-Wno-39029",
            "-Wno-39001",
            "-Wno-40100",
        };
        appendDefineArgs(args, options.defines);
        appendIncludeArgs(args, options);
        appendEntryArgs(args, options);
        args.insert(args.end(), {"-reflection-json", tempFiles.metalProbeJson});
        return args;
    }

    [[nodiscard]] std::string_view viewAsText(std::span<const std::byte> bytes) noexcept
    {
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    struct ReflectionProbe final
    {
        std::string_view description;
        std::vector<std::string> arguments;
        std::string reflectionJsonPath;
        std::string outputLogPath;
    };

    [[nodiscard]] std::string readProbeOutput(const std::string& outputLogPath)
    {
        const auto bytes = readBinaryFile(outputLogPath);
        if (!bytes.has_value())
        {
            return {};
        }
        std::string_view text = viewAsText(*bytes);
        while (!text.empty() &&
               (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '\t'))
        {
            text.remove_suffix(1);
        }
        return std::string(text);
    }

    [[nodiscard]] std::expected<std::vector<std::byte>, std::string> runReflectionProbe(
        const CookMetallibOptions& options, const ReflectionProbe& probe)
    {
        std::error_code ec;
        std::filesystem::remove(pathFromUtf8(probe.reflectionJsonPath), ec);
        const auto exitCode = runProcess(options.slangcPath, probe.arguments, probe.outputLogPath);
        if (!exitCode.has_value())
        {
            return std::unexpected(exitCode.error());
        }
        if (*exitCode != 0)
        {
            std::string message = "slangc " + std::string(probe.description) + " failed with exit code " +
                                  std::to_string(*exitCode) + ": " +
                                  formatCommandLine(options.slangcPath, probe.arguments);
            if (const std::string output = readProbeOutput(probe.outputLogPath); !output.empty())
            {
                message += '\n';
                message += output;
            }
            return std::unexpected(std::move(message));
        }
        auto reflection = readBinaryFile(probe.reflectionJsonPath);
        if (!reflection.has_value())
        {
            return std::unexpected("slangc " + std::string(probe.description) +
                                   " wrote no reflection JSON: " + reflection.error());
        }
        return reflection;
    }

    [[nodiscard]] std::expected<rachis::shader_tool::MetalCookPlan, std::string> probeMetalCookPlan(
        const CookMetallibOptions& options, const CookTempFiles& tempFiles)
    {
        const auto spirvReflection = runReflectionProbe(options,
                                                        ReflectionProbe{
                                                            .description = "SPIR-V reflection probe",
                                                            .arguments = buildSpirvProbeArgs(options, tempFiles),
                                                            .reflectionJsonPath = tempFiles.spirvProbeJson,
                                                            .outputLogPath = tempFiles.spirvProbeLog,
                                                        });
        if (!spirvReflection.has_value())
        {
            return std::unexpected(spirvReflection.error());
        }
        const auto metalReflection = runReflectionProbe(options,
                                                        ReflectionProbe{
                                                            .description = "Metal reflection probe",
                                                            .arguments = buildMetalProbeArgs(options, tempFiles),
                                                            .reflectionJsonPath = tempFiles.metalProbeJson,
                                                            .outputLogPath = tempFiles.metalProbeLog,
                                                        });
        if (!metalReflection.has_value())
        {
            return std::unexpected(metalReflection.error());
        }
        return rachis::shader_tool::planMetalShaderBindings(
            viewAsText(*spirvReflection), viewAsText(*metalReflection), options.entryPoint);
    }

    void appendNonZeroSpaceShift(std::vector<std::string>& args,
                                 std::string_view shiftFlag,
                                 std::uint32_t shift,
                                 std::size_t space)
    {
        if (shift == 0)
        {
            return;
        }
        args.emplace_back(shiftFlag);
        args.push_back(std::to_string(shift));
        args.push_back(std::to_string(space));
    }

    [[nodiscard]] std::vector<std::string> buildMetalCompileArgs(const CookMetallibOptions& options,
                                                                 const rachis::shader_tool::MetalSpaceShifts& shifts,
                                                                 std::string_view target)
    {
        std::vector<std::string> args{
            options.sourcePath,
            "-target",
            std::string(target),
            "-profile",
            "sm_6_0",
            "-matrix-layout-column-major",
            "-restrictive-capability-check",
            "-fvk-use-entrypoint-name",
            "-fvk-b-shift",
            "0",
            "all",
            "-fvk-t-shift",
            "0",
            "all",
            "-fvk-s-shift",
            "0",
            "all",
            "-fvk-u-shift",
            "0",
            "all",
            "-Wno-40100",
        };
        for (std::size_t space = 0; space < rachis::shader_tool::metalShiftSpaceCount; ++space)
        {
            appendNonZeroSpaceShift(args, "-fvk-b-shift", shifts.buffer[space], space);
            appendNonZeroSpaceShift(args, "-fvk-t-shift", shifts.texture[space], space);
            appendNonZeroSpaceShift(args, "-fvk-s-shift", shifts.sampler[space], space);
        }
        if (!options.noWarningsAsErrors)
        {
            args.push_back("-warnings-as-errors");
            args.push_back("all");
        }
        appendDefineArgs(args, options.defines);
        appendIncludeArgs(args, options);
        appendEntryArgs(args, options);
        return args;
    }

    [[nodiscard]] int compileMetallib(const CookMetallibOptions& options,
                                      const rachis::shader_tool::MetalSpaceShifts& shifts,
                                      const std::string& rawReflectionJsonPath)
    {
        auto metallibArgs = buildMetalCompileArgs(options, shifts, "metallib");
        metallibArgs.insert(metallibArgs.end(), {"-reflection-json", rawReflectionJsonPath, "-o", options.outputPath});
        if (options.depfilePath.has_value())
        {
            metallibArgs.insert(metallibArgs.end(), {"-depfile", *options.depfilePath});
        }
        return runCompilerStep(options.slangcPath, metallibArgs);
    }

    [[nodiscard]] int compileDebugMetalSource(const CookMetallibOptions& options,
                                              const rachis::shader_tool::MetalSpaceShifts& shifts,
                                              const std::string& metalSourcePath)
    {
        if (const auto metalParent = pathFromUtf8(metalSourcePath).parent_path(); !metalParent.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(metalParent, ec);
            if (ec)
            {
                std::cerr << "Failed to create debug Metal source directory '" << pathToUtf8(metalParent)
                          << "': " << ec.message() << '\n';
                return 1;
            }
        }
        auto metalSourceArgs = buildMetalCompileArgs(options, shifts, "metal");
        metalSourceArgs.insert(metalSourceArgs.end(), {"-o", metalSourcePath});
        return runCompilerStep(options.slangcPath, metalSourceArgs);
    }

    [[nodiscard]] std::expected<bool, std::string> probeReadsVertexId(const CookTempFiles& tempFiles)
    {
        const auto probeSource = readBinaryFile(tempFiles.metalProbeSource);
        if (!probeSource.has_value())
        {
            return std::unexpected("slangc Metal reflection probe wrote no Metal source: " + probeSource.error());
        }
        const auto rebased = rachis::shader_tool::rebaseMetalVertexIds(viewAsText(*probeSource));
        if (!rebased.has_value())
        {
            return std::unexpected(rebased.error());
        }
        return rebased->has_value();
    }

    [[nodiscard]] std::expected<bool, std::string> probeUsesUnboundedArrays(const CookTempFiles& tempFiles)
    {
        const auto probeSource = readBinaryFile(tempFiles.metalProbeSource);
        if (!probeSource.has_value())
        {
            return std::unexpected("slangc Metal reflection probe wrote no Metal source: " + probeSource.error());
        }
        return !rachis::shader_tool::collectMetalUnboundedArrayParameters(viewAsText(*probeSource)).empty();
    }

    [[nodiscard]] std::vector<std::string> buildMetalCompilerArgs(const CookMetallibOptions& options,
                                                                  const std::string& metalSourcePath)
    {
        std::vector<std::string> args{
            "--sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-fno-strict-aliasing",
            "-Os",
            "-g0",
            "-o",
            options.outputPath,
        };
        appendDefineArgs(args, options.defines);
        args.push_back(metalSourcePath);
        return args;
    }

    [[nodiscard]] int compileRebasedMetallib(const CookMetallibOptions& options,
                                             const rachis::shader_tool::MetalSpaceShifts& shifts,
                                             const CookTempFiles& tempFiles,
                                             std::vector<rachis::shader_tool::MetalArgumentBufferArray>& arrays)
    {
        auto metalSourceArgs = buildMetalCompileArgs(options, shifts, "metal");
        metalSourceArgs.insert(metalSourceArgs.end(),
                               {"-reflection-json", tempFiles.rawReflectionJson, "-o", tempFiles.rebasedMetalSource});
        if (options.depfilePath.has_value())
        {
            metalSourceArgs.insert(metalSourceArgs.end(), {"-depfile", *options.depfilePath});
        }
        if (const int rc = runCompilerStep(options.slangcPath, metalSourceArgs); rc != 0)
        {
            return rc;
        }
        const auto generated = readBinaryFile(tempFiles.rebasedMetalSource);
        if (!generated.has_value())
        {
            std::cerr << generated.error() << '\n';
            return 1;
        }
        const auto rebased = rachis::shader_tool::rebaseMetalVertexIds(viewAsText(*generated));
        if (!rebased.has_value())
        {
            std::cerr << rebased.error() << '\n';
            return 1;
        }
        const std::string rebasedSource = rebased->value_or(std::string(viewAsText(*generated)));
        const auto spirvReflection = readBinaryFile(tempFiles.spirvProbeJson);
        const auto metalReflection = readBinaryFile(tempFiles.rawReflectionJson);
        if (!spirvReflection.has_value() || !metalReflection.has_value())
        {
            std::cerr << "The Metal argument-buffer plan could not read the reflection JSON.\n";
            return 1;
        }
        auto planned = rachis::shader_tool::planMetalArgumentBufferArrays(
            rebasedSource, viewAsText(*spirvReflection), viewAsText(*metalReflection));
        if (!planned.has_value())
        {
            std::cerr << planned.error() << '\n';
            return 1;
        }
        auto rewritten = rachis::shader_tool::rewriteMetalArgumentBufferArrays(rebasedSource, *planned);
        if (!rewritten.has_value())
        {
            std::cerr << rewritten.error() << '\n';
            return 1;
        }
        arrays = std::move(*planned);
        const std::string& source = *rewritten;
        const std::span<const std::byte> sourceBytes(reinterpret_cast<const std::byte*>(source.data()), source.size());
        if (const auto written = writeBinaryFile(tempFiles.rebasedMetalSource, sourceBytes); !written.has_value())
        {
            std::cerr << written.error() << '\n';
            return 1;
        }
        if (options.metalSourcePath.has_value())
        {
            if (const auto written = writeBinaryFile(*options.metalSourcePath, sourceBytes); !written.has_value())
            {
                std::cerr << written.error() << '\n';
                return 1;
            }
        }
        return runCompilerStep("xcrun", buildMetalCompilerArgs(options, tempFiles.rebasedMetalSource));
    }

    [[nodiscard]] std::expected<std::string, std::string> readAndEnrichRawReflection(
        const std::string& rawReflectionJsonPath,
        std::span<const rachis::shader_tool::MetalLogicalParameterBinding> bindings,
        std::string_view entryPoint,
        std::span<const rachis::shader_tool::MetalArgumentBufferArray> argumentBuffers)
    {
        const auto rawReflectionBytes = readBinaryFile(rawReflectionJsonPath);
        if (!rawReflectionBytes.has_value())
        {
            return std::unexpected(rawReflectionBytes.error());
        }
        return rachis::shader_tool::enrichMetalReflectionJson(
            viewAsText(*rawReflectionBytes), bindings, entryPoint, argumentBuffers);
    }

    [[nodiscard]] std::expected<void, std::string> writePnmsManifest(const CookMetallibOptions& options,
                                                                     std::span<const std::byte> reflection)
    {
        const auto metallibBytes = readBinaryFile(options.outputPath);
        if (!metallibBytes.has_value())
        {
            return std::unexpected(metallibBytes.error());
        }
        const auto packedBytes = rachis::shader_tool::packMetalShaderManifest(reflection, *metallibBytes);
        if (!packedBytes.has_value())
        {
            return std::unexpected(packedBytes.error());
        }
        return writeBinaryFile(options.pnmsPath, *packedBytes);
    }

    [[nodiscard]] int cookMetallib(const CookMetallibOptions& options)
    {
        const auto outputFs = prepareOutputPath(options.outputPath);
        if (!outputFs.has_value())
        {
            std::cerr << outputFs.error() << '\n';
            return 1;
        }

        const CookTempFiles tempFiles = makeCookTempFiles(*outputFs);
        ScopedTempFileCleanup tempCleanup;
        scheduleCookTempFileCleanup(tempCleanup, tempFiles);

        const auto cookPlan = probeMetalCookPlan(options, tempFiles);
        if (!cookPlan.has_value())
        {
            std::cerr << cookPlan.error() << '\n';
            return 1;
        }
        const auto readsVertexId = probeReadsVertexId(tempFiles);
        if (!readsVertexId.has_value())
        {
            std::cerr << readsVertexId.error() << '\n';
            return 1;
        }
        const auto usesUnboundedArrays = probeUsesUnboundedArrays(tempFiles);
        if (!usesUnboundedArrays.has_value())
        {
            std::cerr << usesUnboundedArrays.error() << '\n';
            return 1;
        }
        std::vector<rachis::shader_tool::MetalArgumentBufferArray> argumentBuffers;
        if (*readsVertexId || *usesUnboundedArrays)
        {
            if (const int rc = compileRebasedMetallib(options, cookPlan->shifts, tempFiles, argumentBuffers); rc != 0)
            {
                return rc;
            }
        }
        else
        {
            if (const int rc = compileMetallib(options, cookPlan->shifts, tempFiles.rawReflectionJson); rc != 0)
            {
                return rc;
            }
            if (options.metalSourcePath.has_value())
            {
                if (const int rc = compileDebugMetalSource(options, cookPlan->shifts, *options.metalSourcePath);
                    rc != 0)
                {
                    return rc;
                }
            }
        }

        std::vector<rachis::shader_tool::MetalLogicalParameterBinding> bindings;
        for (const auto& binding : cookPlan->bindings)
        {
            if (std::ranges::none_of(argumentBuffers, [&](const auto& array) { return array.name == binding.name; }))
            {
                bindings.push_back(binding);
            }
        }
        const auto enrichedJson =
            readAndEnrichRawReflection(tempFiles.rawReflectionJson, bindings, options.entryPoint, argumentBuffers);
        if (!enrichedJson.has_value())
        {
            std::cerr << enrichedJson.error() << '\n';
            return 1;
        }
        const std::span<const std::byte> enrichedBytes(reinterpret_cast<const std::byte*>(enrichedJson->data()),
                                                       enrichedJson->size());
        if (const auto jsonStatus = writeBinaryFile(options.jsonPath, enrichedBytes); !jsonStatus.has_value())
        {
            std::cerr << jsonStatus.error() << '\n';
            return 1;
        }
        if (const auto pnmsStatus = writePnmsManifest(options, enrichedBytes); !pnmsStatus.has_value())
        {
            std::cerr << pnmsStatus.error() << '\n';
            return 1;
        }
        return 0;
    }

    [[nodiscard]] int runCookMetallib(std::span<const std::string_view> args)
    {
        if (args.size() == 1 && args[0] == "--help")
        {
            std::cout << "--slangc <slangc> --source <shader> --entry <entry> --output <shader.metallib> "
                         "--json <reflection.json> --pnms <shader.pnms> [--stage <stage>] "
                         "[--metal-source <shader.metal>] [--depfile <shader.d>] [--no-warnings-as-errors] "
                         "[--define <NAME=VALUE> ...] [--include-dir <directory> ...]\n";
            return 0;
        }

        const auto options = parseCookMetallibOptions(args);
        if (!options.has_value())
        {
            std::cerr << options.error() << '\n';
            return 1;
        }
        return cookMetallib(*options);
    }

    struct ReflectInterfaceOptions final
    {
        std::string name;
        std::string entryPoint;
        std::string outputPath;
        std::string tag;
        rachis::shader_tool::InterfaceStage stage = rachis::shader_tool::InterfaceStage::compute;
        std::vector<std::pair<rachis::shader_tool::InterfaceBackend, std::string>> reflections;
        std::vector<std::pair<rachis::shader_tool::InterfaceBackend, std::string>> layouts;
        std::vector<std::pair<rachis::shader_tool::InterfaceBackend, std::string>> spirvReflections;
        std::vector<std::pair<rachis::shader_tool::InterfaceBackend, std::string>> bytecodes;
    };

    [[nodiscard]] std::expected<ReflectInterfaceOptions, std::string> parseReflectInterfaceOptions(
        std::span<const std::string_view> args)
    {
        ReflectInterfaceOptions options;
        bool hasStage = false;
        for (std::size_t index = 0; index < args.size(); index += 2)
        {
            if (index + 1 == args.size())
            {
                return std::unexpected("Every option of reflect-interface takes a value.");
            }
            const std::string_view flag = args[index];
            const std::string_view value = args[index + 1];
            if (flag == "--name" && options.name.empty())
            {
                options.name.assign(value);
            }
            else if (flag == "--entry" && options.entryPoint.empty())
            {
                options.entryPoint.assign(value);
            }
            else if (flag == "--output" && options.outputPath.empty())
            {
                options.outputPath.assign(value);
            }
            else if (flag == "--tag" && options.tag.empty())
            {
                options.tag.assign(value);
            }
            else if (flag == "--stage" && !hasStage)
            {
                const auto stage = rachis::shader_tool::parseInterfaceStage(value);
                if (!stage.has_value())
                {
                    return std::unexpected("--stage must be vertex, pixel, or compute.");
                }
                options.stage = *stage;
                hasStage = true;
            }
            else if (flag == "--layout")
            {
                const auto separator = value.find('=');
                const auto backend = separator == std::string_view::npos
                                         ? std::nullopt
                                         : rachis::shader_tool::parseInterfaceBackend(value.substr(0, separator));
                if (!backend.has_value() || separator + 1 == value.size())
                {
                    return std::unexpected("--layout takes <d3d12|vulkan|metal>=<file>.");
                }
                options.layouts.emplace_back(*backend, std::string(value.substr(separator + 1)));
            }
            else if (flag == "--reflection")
            {
                const auto separator = value.find('=');
                const auto backend = separator == std::string_view::npos
                                         ? std::nullopt
                                         : rachis::shader_tool::parseInterfaceBackend(value.substr(0, separator));
                if (!backend.has_value() || separator + 1 == value.size())
                {
                    return std::unexpected("--reflection takes <d3d12|vulkan|metal>=<reflection.json>.");
                }
                for (const auto& item : options.reflections)
                {
                    if (item.first == *backend)
                    {
                        return std::unexpected("--reflection names a backend twice.");
                    }
                }
                options.reflections.emplace_back(*backend, std::string(value.substr(separator + 1)));
            }
            else if (flag == "--bytecode")
            {
                const auto separator = value.find('=');
                const auto backend = separator == std::string_view::npos
                                         ? std::nullopt
                                         : rachis::shader_tool::parseInterfaceBackend(value.substr(0, separator));
                if (!backend.has_value() || *backend == rachis::shader_tool::InterfaceBackend::metal ||
                    separator + 1 == value.size())
                {
                    return std::unexpected("--bytecode takes <d3d12|vulkan>=<file>.");
                }
                if (std::ranges::any_of(options.bytecodes, [&](const auto& item) { return item.first == *backend; }))
                {
                    return std::unexpected("--bytecode names a backend twice.");
                }
                options.bytecodes.emplace_back(*backend, std::string(value.substr(separator + 1)));
            }
            else if (flag == "--spirv-reflection")
            {
                const auto separator = value.find('=');
                const auto backend = separator == std::string_view::npos
                                         ? std::nullopt
                                         : rachis::shader_tool::parseInterfaceBackend(value.substr(0, separator));
                if (backend != rachis::shader_tool::InterfaceBackend::vulkan || separator + 1 == value.size())
                {
                    return std::unexpected("--spirv-reflection takes vulkan=<module.spv>.");
                }
                options.spirvReflections.emplace_back(*backend, std::string(value.substr(separator + 1)));
            }
            else
            {
                return std::unexpected("Unknown or repeated option: " + std::string(flag));
            }
        }
        if (options.name.empty() || options.entryPoint.empty() || options.outputPath.empty() || !hasStage ||
            (options.reflections.empty() && options.spirvReflections.empty()))
        {
            return std::unexpected("--name, --stage, --entry, --output, and at least one --reflection or "
                                   "--spirv-reflection are required.");
        }
        const auto namesVulkan = [](const auto& items) {
            return std::ranges::any_of(
                items, [](const auto& item) { return item.first == rachis::shader_tool::InterfaceBackend::vulkan; });
        };
        if (options.spirvReflections.size() > 1 ||
            (!options.spirvReflections.empty() && (namesVulkan(options.reflections) || namesVulkan(options.layouts))))
        {
            return std::unexpected("--spirv-reflection reflects Vulkan alone, once, without a --reflection or "
                                   "--layout for Vulkan.");
        }
        return options;
    }

    [[nodiscard]] int runReflectInterface(std::span<const std::string_view> args)
    {
        if (args.size() == 1 && args[0] == "--help")
        {
            std::cout << "--name <identifier> --stage <stage> --entry <entry> --output <header> "
                         "--reflection <backend>=<reflection.json> ... [--layout <backend>=<file> ...] "
                         "[--spirv-reflection vulkan=<module.spv>] [--bytecode <d3d12|vulkan>=<file> ...] "
                         "[--tag <tag>]\n";
            return 0;
        }
        const auto options = parseReflectInterfaceOptions(args);
        if (!options.has_value())
        {
            std::cerr << options.error() << '\n';
            return 1;
        }
        std::vector<rachis::shader_tool::BackendInterface> backends;
        for (const auto& [backend, path] : options->reflections)
        {
            const auto bytes = readBinaryFile(path);
            if (!bytes.has_value())
            {
                std::cerr << bytes.error() << '\n';
                return 1;
            }
            auto reflected = rachis::shader_tool::reflectShaderInterface(
                viewAsText(*bytes), backend, options->entryPoint, options->stage);
            if (!reflected.has_value())
            {
                std::cerr << path << ": " << reflected.error() << '\n';
                return 1;
            }
            std::optional<std::string> layoutPath;
            for (const auto& [layoutBackend, file] : options->layouts)
            {
                if (layoutBackend == backend)
                {
                    layoutPath = file;
                }
            }
            if (layoutPath.has_value() || backend == rachis::shader_tool::InterfaceBackend::d3d12)
            {
                const auto layoutBytes = layoutPath.has_value() ? readBinaryFile(*layoutPath) : bytes;
                if (!layoutBytes.has_value())
                {
                    std::cerr << layoutBytes.error() << '\n';
                    return 1;
                }
                const auto attached = backend == rachis::shader_tool::InterfaceBackend::vulkan
                                          ? rachis::shader_tool::attachSpirvDataLayouts(*reflected, *layoutBytes)
                                          : rachis::shader_tool::attachSlangDataLayouts(
                                                *reflected,
                                                viewAsText(*layoutBytes),
                                                backend == rachis::shader_tool::InterfaceBackend::metal);
                if (!attached.has_value())
                {
                    std::cerr << layoutPath.value_or(path) << ": " << attached.error() << '\n';
                    return 1;
                }
                if (backend == rachis::shader_tool::InterfaceBackend::vulkan)
                {
                    const auto drawIndex =
                        rachis::shader_tool::spirvReadsDrawIndex(*layoutBytes, options->entryPoint, options->stage);
                    if (!drawIndex.has_value())
                    {
                        std::cerr << *layoutPath << ": " << drawIndex.error() << '\n';
                        return 1;
                    }
                    reflected->drawIndex = *drawIndex;
                }
            }
            backends.push_back({.backend = backend, .reflected = std::move(*reflected)});
        }
        for (const auto& [backend, path] : options->spirvReflections)
        {
            const auto bytes = readBinaryFile(path);
            if (!bytes.has_value())
            {
                std::cerr << bytes.error() << '\n';
                return 1;
            }
            auto reflected =
                rachis::shader_tool::reflectSpirvShaderInterface(*bytes, options->entryPoint, options->stage);
            if (!reflected.has_value())
            {
                std::cerr << path << ": " << reflected.error() << '\n';
                return 1;
            }
            backends.push_back({.backend = backend, .reflected = std::move(*reflected)});
        }
        for (const auto& [backend, path] : options->bytecodes)
        {
            auto* const target = [&]() -> rachis::shader_tool::BackendInterface* {
                for (auto& item : backends)
                {
                    if (item.backend == backend)
                    {
                        return &item;
                    }
                }
                return nullptr;
            }();
            if (target == nullptr)
            {
                std::cerr << "--bytecode names a backend that no --reflection or --spirv-reflection reflects.\n";
                return 1;
            }
            const auto bytes = readBinaryFile(path);
            if (!bytes.has_value())
            {
                std::cerr << bytes.error() << '\n';
                return 1;
            }
            const auto drawIndex =
                backend == rachis::shader_tool::InterfaceBackend::d3d12
                    ? rachis::shader_tool::dxilReadsDrawIndex(*bytes)
                    : rachis::shader_tool::spirvReadsDrawIndex(*bytes, options->entryPoint, options->stage);
            if (!drawIndex.has_value())
            {
                std::cerr << path << ": " << drawIndex.error() << '\n';
                return 1;
            }
            target->reflected.drawIndex = *drawIndex && options->stage == rachis::shader_tool::InterfaceStage::vertex;
        }
        const auto header = rachis::shader_tool::generateShaderInterfaceHeader(
            options->name, options->stage, options->entryPoint, backends, options->tag);
        if (!header.has_value())
        {
            std::cerr << header.error() << '\n';
            return 1;
        }
        const std::span<const std::byte> headerBytes(reinterpret_cast<const std::byte*>(header->data()),
                                                     header->size());
        if (const auto written = writeBinaryFile(options->outputPath, headerBytes); !written.has_value())
        {
            std::cerr << written.error() << '\n';
            return 1;
        }
        return 0;
    }

    [[nodiscard]] int runTool(std::span<const std::string_view> args)
    {
        if (args.empty())
        {
            printUsage(std::cerr);
            return 1;
        }

        const std::string_view subcommand = args.front();
        if (subcommand == "--help")
        {
            if (args.size() != 1)
            {
                printUsage(std::cerr);
                return 1;
            }
            printUsage(std::cout);
            return 0;
        }
        if (subcommand == "embed-header")
        {
            return runEmbedHeader(args.subspan(1));
        }
        if (subcommand == "pack-pnms")
        {
            return runPackPnms(args.subspan(1));
        }
        if (subcommand == "cook-metallib")
        {
            return runCookMetallib(args.subspan(1));
        }
        if (subcommand == "reflect-interface")
        {
            return runReflectInterface(args.subspan(1));
        }

        std::cerr << "Unknown subcommand: " << subcommand << '\n';
        printUsage(std::cerr);
        return 1;
    }
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[])
{
    std::vector<std::string> storage;
    storage.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0U);
    for (int index = 1; index < argc; ++index)
    {
        storage.push_back(narrowToUtf8(argv[index]));
    }
    const std::vector<std::string_view> args(storage.begin(), storage.end());
    return runTool(args);
}
#else
int main(int argc, char* argv[])
{
    const std::vector<std::string_view> args(argv + (argc > 0 ? 1 : 0), argv + argc);
    return runTool(args);
}
#endif
