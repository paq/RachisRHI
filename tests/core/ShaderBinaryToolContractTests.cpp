#include "Internal/ShaderBinaryTool.h"

#if RACHIS_ENABLE_METAL
#include "Internal/MetalShaderManifest.h"
#include "Metal/MetalTestHooks.h"
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <clocale>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace
{

    [[nodiscard]] std::span<const std::byte> asBytes(std::string_view text) noexcept
    {
        return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
    }

    [[nodiscard]] std::uint32_t readU32LittleEndian(std::span<const std::byte> bytes, std::size_t offset) noexcept
    {
        return static_cast<std::uint32_t>(std::to_integer<unsigned int>(bytes[offset])) |
               (static_cast<std::uint32_t>(std::to_integer<unsigned int>(bytes[offset + 1])) << 8U) |
               (static_cast<std::uint32_t>(std::to_integer<unsigned int>(bytes[offset + 2])) << 16U) |
               (static_cast<std::uint32_t>(std::to_integer<unsigned int>(bytes[offset + 3])) << 24U);
    }

    [[nodiscard]] std::string readTextFile(const std::filesystem::path& path)
    {
        std::ifstream const stream(path, std::ios::binary);
        REQUIRE(stream);
        std::ostringstream buffer;
        buffer << stream.rdbuf();
        return buffer.str();
    }

    [[nodiscard]] std::vector<std::byte> readBinaryFile(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        REQUIRE(stream);
        const auto endPos = stream.tellg();
        REQUIRE(endPos >= 0);
        const auto size = static_cast<std::size_t>(endPos);
        stream.seekg(0, std::ios::beg);
        std::vector<std::byte> bytes(size);
        if (size > 0)
        {
            stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
            REQUIRE(stream);
        }
        return bytes;
    }

    void writeBinaryFile(const std::filesystem::path& path, std::span<const std::byte> data)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        REQUIRE(stream);
        if (!data.empty())
        {
            stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
        stream.flush();
        REQUIRE(stream);
    }

    struct ScopedTempDirectory final
    {
        std::filesystem::path path;

        ScopedTempDirectory()
        {
#ifdef _WIN32
            const auto pid = static_cast<unsigned long>(_getpid());
#else
            const auto pid = static_cast<unsigned long>(getpid());
#endif
            path = std::filesystem::temp_directory_path() / ("rachis-shader-binary-tool-" + std::to_string(pid));
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path);
        }

        ~ScopedTempDirectory()
        {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        ScopedTempDirectory(const ScopedTempDirectory&) = delete;
        ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;
    };

#ifdef RACHIS_SHADER_BINARY_TOOL_PATH
    struct CliRunResult final
    {
        int status = -1;
        std::string stderrText;
    };

    using NativeString = std::filesystem::path::string_type;

    void appendAscii(NativeString& command, std::string_view text)
    {
        for (const char ch : text)
        {
            command.push_back(static_cast<std::filesystem::path::value_type>(ch));
        }
    }

    void appendQuotedShellArg(NativeString& command, const NativeString& arg)
    {
        command.push_back('"');
        for (const auto ch : arg)
        {
            if (ch == '"')
            {
                command.push_back('\\');
            }
            command.push_back(ch);
        }
        command.push_back('"');
    }

    [[nodiscard]] CliRunResult runShaderBinaryTool(const std::filesystem::path& scratchDir,
                                                   std::initializer_list<std::filesystem::path> args)
    {
        const auto tool = std::filesystem::path(RACHIS_SHADER_BINARY_TOOL_PATH).lexically_normal();
        REQUIRE(std::filesystem::is_regular_file(tool));

        const auto stdoutPath = scratchDir / "cli-stdout.txt";
        const auto stderrPath = scratchDir / "cli-stderr.txt";
        std::error_code ec;
        std::filesystem::remove(stdoutPath, ec);
        std::filesystem::remove(stderrPath, ec);

        NativeString command;
        appendQuotedShellArg(command, tool.native());
        for (const auto& arg : args)
        {
            command.push_back(' ');
            appendQuotedShellArg(command, arg.native());
        }
        appendAscii(command, " > ");
        appendQuotedShellArg(command, stdoutPath.native());
        appendAscii(command, " 2> ");
        appendQuotedShellArg(command, stderrPath.native());
#ifdef _WIN32
        command.insert(command.begin(), '"');
        command.push_back('"');
        const int status = _wsystem(command.c_str());
#else
        const int status = std::system(command.c_str());
#endif
        return CliRunResult{
            .status = status,
            .stderrText = std::filesystem::exists(stderrPath) ? readTextFile(stderrPath) : std::string{},
        };
    }

#ifdef _WIN32
    [[nodiscard]] bool activeCodePageRepresents(wchar_t character)
    {
        std::array<char, 8> narrow{};
        BOOL usedDefaultChar = FALSE;
        const int narrowSize = WideCharToMultiByte(CP_ACP,
                                                   WC_NO_BEST_FIT_CHARS,
                                                   &character,
                                                   1,
                                                   narrow.data(),
                                                   static_cast<int>(narrow.size()),
                                                   nullptr,
                                                   &usedDefaultChar);
        if (narrowSize <= 0 || usedDefaultChar != FALSE)
        {
            return false;
        }
        wchar_t roundTrip = L'\0';
        return MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, narrow.data(), narrowSize, &roundTrip, 1) == 1 &&
               roundTrip == character;
    }

    [[nodiscard]] std::optional<wchar_t> findCharacterOutsideActiveCodePage()
    {
        for (const wchar_t candidate : {L'é', L'Ł', L'ж', L'ა'})
        {
            if (!activeCodePageRepresents(candidate))
            {
                return candidate;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::string utf8Name(const std::filesystem::path& path)
    {
        const std::u8string name = path.filename().u8string();
        return std::string(name.begin(), name.end());
    }

    [[nodiscard]] CliRunResult runShaderBinaryToolProcess(const std::filesystem::path& scratchDir,
                                                          const std::filesystem::path& workingDir,
                                                          std::initializer_list<std::filesystem::path> args)
    {
        const auto tool = std::filesystem::path(RACHIS_SHADER_BINARY_TOOL_PATH).lexically_normal();
        REQUIRE(std::filesystem::is_regular_file(tool));

        NativeString commandLine;
        appendQuotedShellArg(commandLine, tool.native());
        for (const auto& arg : args)
        {
            commandLine.push_back(L' ');
            appendQuotedShellArg(commandLine, arg.native());
        }

        const auto outputPath = scratchDir / "cli-process-output.txt";
        SECURITY_ATTRIBUTES inheritable{};
        inheritable.nLength = sizeof(inheritable);
        inheritable.bInheritHandle = TRUE;
        HANDLE const outputHandle = CreateFileW(outputPath.c_str(),
                                                GENERIC_WRITE,
                                                FILE_SHARE_READ,
                                                &inheritable,
                                                CREATE_ALWAYS,
                                                FILE_ATTRIBUTE_NORMAL,
                                                nullptr);
        REQUIRE(outputHandle != INVALID_HANDLE_VALUE);

        STARTUPINFOW startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        startupInfo.dwFlags = STARTF_USESTDHANDLES;
        startupInfo.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startupInfo.hStdOutput = outputHandle;
        startupInfo.hStdError = outputHandle;
        PROCESS_INFORMATION processInfo{};
        const BOOL created = CreateProcessW(nullptr,
                                            commandLine.data(),
                                            nullptr,
                                            nullptr,
                                            TRUE,
                                            0,
                                            nullptr,
                                            workingDir.c_str(),
                                            &startupInfo,
                                            &processInfo);
        const DWORD createError = created != FALSE ? ERROR_SUCCESS : GetLastError();
        CloseHandle(outputHandle);
        INFO("CreateProcessW GetLastError=" << createError);
        REQUIRE(created != FALSE);
        CloseHandle(processInfo.hThread);

        const bool exited = WaitForSingleObject(processInfo.hProcess, 20000) == WAIT_OBJECT_0;
        if (!exited)
        {
            TerminateProcess(processInfo.hProcess, 1);
            WaitForSingleObject(processInfo.hProcess, INFINITE);
        }
        DWORD exitCode = 0;
        const bool gotExitCode = GetExitCodeProcess(processInfo.hProcess, &exitCode) != FALSE;
        CloseHandle(processInfo.hProcess);
        REQUIRE(exited);
        REQUIRE(gotExitCode);
        return CliRunResult{
            .status = static_cast<int>(exitCode),
            .stderrText = readTextFile(outputPath),
        };
    }
#endif

#ifdef RACHIS_FAKE_SLANGC_PATH
    struct CookOutputs final
    {
        std::filesystem::path metallib;
        std::filesystem::path json;
        std::filesystem::path pnms;
    };

    [[nodiscard]] CookOutputs cookOutputsIn(const std::filesystem::path& directory)
    {
        return CookOutputs{
            .metallib = directory / "shader.metallib",
            .json = directory / "shader.metallib.json",
            .pnms = directory / "shader.pnms",
        };
    }

    [[nodiscard]] bool anyCookOutputExists(const CookOutputs& outputs)
    {
        return std::filesystem::exists(outputs.metallib) || std::filesystem::exists(outputs.json) ||
               std::filesystem::exists(outputs.pnms);
    }

    [[nodiscard]] CliRunResult runCookMetallib(const std::filesystem::path& scratchDir,
                                               const std::filesystem::path& slangc,
                                               const std::filesystem::path& source,
                                               const CookOutputs& outputs,
                                               const std::filesystem::path& define)
    {
        return runShaderBinaryTool(scratchDir,
                                   {"cook-metallib",
                                    "--slangc",
                                    slangc,
                                    "--source",
                                    source,
                                    "--entry",
                                    "computeMain",
                                    "--output",
                                    outputs.metallib,
                                    "--json",
                                    outputs.json,
                                    "--pnms",
                                    outputs.pnms,
                                    "--define",
                                    define});
    }

    constexpr std::string_view fakeCookSource = "[shader(\"compute\")] void computeMain() {}\n";
    constexpr std::string_view fakeCookPushBinding = R"({"name": "pushBlock", "pushConstant": true})";
    constexpr std::string_view fakeCookOutputBinding = R"({"name": "output", "set": 1, "binding": 3})";
#endif
#endif

    class ScopedCommaDecimalNumericLocale final
    {
    public:
        ScopedCommaDecimalNumericLocale()
        {
            if (const char* current = std::setlocale(LC_NUMERIC, nullptr); current != nullptr)
            {
                previous_ = current;
            }
            for (const char* name :
                 {"de-DE", "German_Germany.1252", "de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"})
            {
                if (std::setlocale(LC_NUMERIC, name) != nullptr &&
                    std::string_view{std::localeconv()->decimal_point} == ",")
                {
                    active_ = true;
                    return;
                }
            }
            std::setlocale(LC_NUMERIC, previous_.c_str());
        }

        ~ScopedCommaDecimalNumericLocale()
        {
            std::setlocale(LC_NUMERIC, previous_.c_str());
        }

        ScopedCommaDecimalNumericLocale(const ScopedCommaDecimalNumericLocale&) = delete;
        ScopedCommaDecimalNumericLocale& operator=(const ScopedCommaDecimalNumericLocale&) = delete;

        [[nodiscard]] bool active() const noexcept
        {
            return active_;
        }

    private:
        std::string previous_ = "C";
        bool active_ = false;
    };

} // namespace

TEST_CASE("ShaderBinaryTool embeds binary spans into deterministic C++ headers", "[size:small][rhi][shader][tooling]")
{
    const auto emptyOk = rachis::shader_tool::generateEmbeddedBinaryHeader({}, "_ok_1");
    REQUIRE(emptyOk.has_value());
    CHECK(*emptyOk == "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
                      "inline constexpr std::uint8_t _ok_1[] = {\n"
                      "    0,\n"
                      "};\n"
                      "inline constexpr std::size_t _ok_1Size = 0;\n");

    constexpr std::array<std::byte, 1> singleByte{std::byte{0x0a}};
    const auto singleHeader = rachis::shader_tool::generateEmbeddedBinaryHeader(singleByte, "g_rachisTest");
    REQUIRE(singleHeader.has_value());
    CHECK(*singleHeader == "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
                           "inline constexpr std::uint8_t g_rachisTest[] = {\n"
                           "    0x0a,\n"
                           "};\n"
                           "inline constexpr std::size_t g_rachisTestSize = sizeof(g_rachisTest);\n");

    std::array<std::byte, 16> sixteenBytes{};
    for (std::size_t index = 0; index < sixteenBytes.size(); ++index)
    {
        sixteenBytes[index] = static_cast<std::byte>(index);
    }
    const auto sixteenHeader = rachis::shader_tool::generateEmbeddedBinaryHeader(sixteenBytes, "g_rachisTest");
    REQUIRE(sixteenHeader.has_value());
    CHECK(*sixteenHeader ==
          "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
          "inline constexpr std::uint8_t g_rachisTest[] = {\n"
          "    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,\n"
          "};\n"
          "inline constexpr std::size_t g_rachisTestSize = sizeof(g_rachisTest);\n");

    std::array<std::byte, 17> seventeenBytes{};
    std::ranges::copy(sixteenBytes, seventeenBytes.begin());
    seventeenBytes[16] = std::byte{0xff};
    const auto seventeenHeader = rachis::shader_tool::generateEmbeddedBinaryHeader(seventeenBytes, "_ok_1");
    REQUIRE(seventeenHeader.has_value());
    CHECK(*seventeenHeader ==
          "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
          "inline constexpr std::uint8_t _ok_1[] = {\n"
          "    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,\n"
          "    0xff,\n"
          "};\n"
          "inline constexpr std::size_t _ok_1Size = sizeof(_ok_1);\n");

    for (const std::string_view invalid :
         {std::string_view{""}, std::string_view{"9bad"}, std::string_view{"bad-name"}, std::string_view{"bad name"}})
    {
        const auto rejected = rachis::shader_tool::generateEmbeddedBinaryHeader(singleByte, invalid);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error() == "The symbol must be a C++ identifier.");
    }
}

TEST_CASE("ShaderBinaryTool packs PNMS manifests and rejects invalid metallib headers",
          "[size:small][rhi][shader][tooling]")
{
    constexpr std::string_view reflectionText = R"({"entryPoints":[{"name":"vertexMain","stage":"vertex"}]})";
    constexpr std::array<std::byte, 8> metallibBytes{
        std::byte{'M'},
        std::byte{'T'},
        std::byte{'L'},
        std::byte{'B'},
        std::byte{0x01},
        std::byte{0x02},
        std::byte{0x03},
        std::byte{0x04},
    };

    const auto packed = rachis::shader_tool::packMetalShaderManifest(asBytes(reflectionText), metallibBytes);
    REQUIRE(packed.has_value());
    REQUIRE(packed->size() == 16 + reflectionText.size() + metallibBytes.size());
    CHECK((*packed)[0] == std::byte{'P'});
    CHECK((*packed)[1] == std::byte{'N'});
    CHECK((*packed)[2] == std::byte{'M'});
    CHECK((*packed)[3] == std::byte{'S'});
    CHECK(readU32LittleEndian(*packed, 4) == 1U);
    CHECK(readU32LittleEndian(*packed, 8) == static_cast<std::uint32_t>(reflectionText.size()));
    CHECK(readU32LittleEndian(*packed, 12) == static_cast<std::uint32_t>(metallibBytes.size()));
    CHECK(std::ranges::equal(std::span<const std::byte>{*packed}.subspan(16, reflectionText.size()),
                             asBytes(reflectionText)));
    CHECK(std::ranges::equal(
        std::span<const std::byte>{*packed}.subspan(16 + reflectionText.size(), metallibBytes.size()), metallibBytes));

    const auto emptyReflection = rachis::shader_tool::packMetalShaderManifest({}, metallibBytes);
    REQUIRE(emptyReflection.has_value());
    REQUIRE(emptyReflection->size() == 16 + metallibBytes.size());
    CHECK(readU32LittleEndian(*emptyReflection, 4) == 1U);
    CHECK(readU32LittleEndian(*emptyReflection, 8) == 0U);
    CHECK(readU32LittleEndian(*emptyReflection, 12) == static_cast<std::uint32_t>(metallibBytes.size()));
    CHECK(std::ranges::equal(std::span<const std::byte>{*emptyReflection}.subspan(16), metallibBytes));

    for (const std::string_view badMagic :
         {std::string_view{""}, std::string_view{"MTL"}, std::string_view{"XXXX"}, std::string_view{"mtlb0000"}})
    {
        const auto rejected = rachis::shader_tool::packMetalShaderManifest(asBytes(reflectionText), asBytes(badMagic));
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error() == "metallib does not start with MTLB.");
    }

#if RACHIS_ENABLE_METAL && defined(RACHIS_METALLIB_DIR)
    const auto metallibDir = std::filesystem::path(RACHIS_METALLIB_DIR);
    const auto cookedJson = readBinaryFile(metallibDir / "BootstrapTriangle.vs.metallib.json");
    const auto cookedMetallib = readBinaryFile(metallibDir / "BootstrapTriangle.vs.metallib");
    const auto cookedPnms = readBinaryFile(metallibDir / "BootstrapTriangle.vs.pnms");

    const auto repacked = rachis::shader_tool::packMetalShaderManifest(cookedJson, cookedMetallib);
    REQUIRE(repacked.has_value());
    CHECK(*repacked == cookedPnms);

    const auto unpacked = rachis::metal::unpackShaderManifest(*repacked);
    REQUIRE(unpacked.has_value());
    CHECK(std::ranges::equal(unpacked->json, cookedJson));
    CHECK(std::ranges::equal(unpacked->metallib, cookedMetallib));

    const std::string_view jsonView(reinterpret_cast<const char*>(unpacked->json.data()), unpacked->json.size());
    const auto reflected = rachis::metal::parseSlangReflectionJson(jsonView, "vertexMain", rachis::ShaderStage::vertex);
    REQUIRE(reflected.has_value());
    REQUIRE(reflected->stageInterface.has_value());
    CHECK(reflected->stageInterface->inputs.empty());
    const std::vector<rachis::detail::StageInterfaceVariable> expectedOutputs{
        {.location = 0, .componentCount = 3, .componentType = rachis::detail::StageComponentType::float32},
    };
    CHECK(reflected->stageInterface->outputs == expectedOutputs);
#endif
}

#if RACHIS_ENABLE_METAL && defined(RACHIS_METALLIB_DIR)
TEST_CASE("Metal GPU accepts metallib unpacked from ShaderBinaryTool PNMS manifest",
          "[size:large][rhi][metal][shader][tooling]")
{
    const auto metallibDir = std::filesystem::path(RACHIS_METALLIB_DIR);
    const auto cookedJson = readBinaryFile(metallibDir / "BootstrapTriangle.vs.metallib.json");
    const auto cookedMetallib = readBinaryFile(metallibDir / "BootstrapTriangle.vs.metallib");

    const auto repacked = rachis::shader_tool::packMetalShaderManifest(cookedJson, cookedMetallib);
    REQUIRE(repacked.has_value());
    const auto unpacked = rachis::metal::unpackShaderManifest(*repacked);
    REQUIRE(unpacked.has_value());
    const auto status = rachis::metal::loadCookedMetallib(unpacked->metallib, "vertexMain");
    REQUIRE(status.has_value());
}
#endif

#ifdef RACHIS_SHADER_BINARY_TOOL_PATH
TEST_CASE("RachisShaderBinaryTool CLI validates flags and writes binary artifacts",
          "[size:small][rhi][shader][tooling]")
{
    ScopedTempDirectory const temp;

    const auto inputBin = temp.path / "sample.bin";
    constexpr std::array<std::byte, 4> sampleBytes{std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}};
    writeBinaryFile(inputBin, sampleBytes);

    const auto outputHeader = temp.path / "nested" / "headers" / "Sample.h";
    const auto embedResult = runShaderBinaryTool(
        temp.path,
        {"embed-header", "--input", inputBin.string(), "--output", outputHeader.string(), "--symbol", "g_rachisTest"});
    INFO("stderr=" << embedResult.stderrText);
    REQUIRE(embedResult.status == 0);
    REQUIRE(std::filesystem::is_regular_file(outputHeader));
    const auto generatedHeader = readTextFile(outputHeader);
    CHECK_FALSE(generatedHeader.contains('\r'));
    const auto expectedHeader = rachis::shader_tool::generateEmbeddedBinaryHeader(sampleBytes, "g_rachisTest");
    REQUIRE(expectedHeader.has_value());
    CHECK(generatedHeader == *expectedHeader);

    const auto jsonPath = temp.path / "reflection.json";
    const auto metallibPath = temp.path / "sample.metallib";
    constexpr std::string_view jsonPayload = R"({"parameters":[]})";
    constexpr std::array<std::byte, 6> metallibPayload{
        std::byte{'M'}, std::byte{'T'}, std::byte{'L'}, std::byte{'B'}, std::byte{0xaa}, std::byte{0x55}};
    writeBinaryFile(jsonPath, asBytes(jsonPayload));
    writeBinaryFile(metallibPath, metallibPayload);

    const auto outputPnms = temp.path / "nested" / "pnms" / "sample.pnms";
    const auto packResult = runShaderBinaryTool(temp.path,
                                                {"pack-pnms",
                                                 "--json",
                                                 jsonPath.string(),
                                                 "--metallib",
                                                 metallibPath.string(),
                                                 "--output",
                                                 outputPnms.string()});
    INFO("stderr=" << packResult.stderrText);
    REQUIRE(packResult.status == 0);
    REQUIRE(std::filesystem::is_regular_file(outputPnms));
    const auto expectedPnms = rachis::shader_tool::packMetalShaderManifest(asBytes(jsonPayload), metallibPayload);
    REQUIRE(expectedPnms.has_value());
    CHECK(readBinaryFile(outputPnms) == *expectedPnms);

    CHECK(runShaderBinaryTool(temp.path, {}).status != 0);
    CHECK(runShaderBinaryTool(temp.path, {"unknown-subcommand"}).status != 0);
    CHECK(runShaderBinaryTool(temp.path,
                              {"embed-header", "--input", inputBin.string(), "--output", outputHeader.string()})
              .status != 0);
    CHECK(runShaderBinaryTool(temp.path,
                              {"embed-header",
                               "--input",
                               inputBin.string(),
                               "--input",
                               inputBin.string(),
                               "--output",
                               outputHeader.string(),
                               "--symbol",
                               "g_ok"})
              .status != 0);

    const auto badSymbol = runShaderBinaryTool(
        temp.path,
        {"embed-header", "--input", inputBin.string(), "--output", outputHeader.string(), "--symbol", "9bad"});
    CHECK(badSymbol.status != 0);
    CHECK(badSymbol.stderrText.contains("The symbol must be a C++ identifier."));

    const auto badMetallibPath = temp.path / "bad.metallib";
    writeBinaryFile(badMetallibPath, asBytes("XXXX"));
    const auto badPack = runShaderBinaryTool(temp.path,
                                             {"pack-pnms",
                                              "--json",
                                              jsonPath.string(),
                                              "--metallib",
                                              badMetallibPath.string(),
                                              "--output",
                                              (temp.path / "bad.pnms").string()});
    CHECK(badPack.status != 0);
    CHECK(badPack.stderrText.contains("metallib does not start with MTLB."));

    CHECK(runShaderBinaryTool(temp.path, {"cook-metallib", "--entry", "computeMain"}).status != 0);
}

#ifdef _WIN32
TEST_CASE("RachisShaderBinaryTool reads arguments outside the active code page through UTF-16",
          "[size:small][rhi][shader][tooling]")
{
    if (GetACP() == CP_UTF8)
    {
        SKIP("The active code page is UTF-8, so no path character is lost.");
    }
    const auto character = findCharacterOutsideActiveCodePage();
    if (!character.has_value())
    {
        SKIP("The active code page represents every candidate character.");
    }

    ScopedTempDirectory const temp;
    const auto workingDir = temp.path / "work";
    const auto directoryName = std::filesystem::path(std::wstring(L"uni-") + *character);
    const auto directory = workingDir / directoryName;
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    REQUIRE_FALSE(ec);
    constexpr std::array<std::byte, 4> sampleBytes{std::byte{0xca}, std::byte{0xfe}, std::byte{0xba}, std::byte{0xbe}};
    writeBinaryFile(directory / "in.bin", sampleBytes);

    const auto result = runShaderBinaryToolProcess(temp.path,
                                                   workingDir,
                                                   {"embed-header",
                                                    "--input",
                                                    directoryName / "in.bin",
                                                    "--output",
                                                    directoryName / "out.h",
                                                    "--symbol",
                                                    "g_uni"});
    INFO("stderr=" << result.stderrText);
    CHECK(result.status == 0);

    std::vector<std::string> workingEntries;
    for (const auto& entry : std::filesystem::directory_iterator(workingDir))
    {
        workingEntries.push_back(utf8Name(entry.path()));
    }
    CHECK(workingEntries == std::vector<std::string>{utf8Name(directoryName)});
    std::vector<std::string> directoryEntries;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        directoryEntries.push_back(utf8Name(entry.path()));
    }
    std::ranges::sort(directoryEntries);
    CHECK(directoryEntries == std::vector<std::string>{"in.bin", "out.h"});

    REQUIRE(std::filesystem::is_regular_file(directory / "out.h"));
    const auto expectedHeader = rachis::shader_tool::generateEmbeddedBinaryHeader(sampleBytes, "g_uni");
    REQUIRE(expectedHeader.has_value());
    CHECK(readTextFile(directory / "out.h") == *expectedHeader);
}
#endif

#ifdef RACHIS_FAKE_SLANGC_PATH
TEST_CASE("RachisShaderBinaryTool cook-metallib stops when a slangc reflection probe fails",
          "[size:small][rhi][shader][tooling]")
{
    ScopedTempDirectory const temp;
    const auto slangc = std::filesystem::path(RACHIS_FAKE_SLANGC_PATH).lexically_normal();
    REQUIRE(std::filesystem::is_regular_file(slangc));
    const auto source = temp.path / "shader.slang";
    writeBinaryFile(source, asBytes(fakeCookSource));

    const auto cooked = cookOutputsIn(temp.path / "cooked");
    const auto cookResult = runCookMetallib(temp.path, slangc, source, cooked, "RACHIS_METAL=1");
    INFO("stderr=" << cookResult.stderrText);
    REQUIRE(cookResult.status == 0);
    CHECK(cookResult.stderrText.contains("fake slangc: metallib ran"));
    CHECK_FALSE(cookResult.stderrText.contains("spirv-probe"));
    CHECK_FALSE(cookResult.stderrText.contains("metal-probe"));
    const auto cookedJson = readTextFile(cooked.json);
    CHECK(cookedJson.contains(fakeCookPushBinding));
    CHECK(cookedJson.contains(fakeCookOutputBinding));
    CHECK(std::filesystem::is_regular_file(cooked.pnms));

    struct BrokenProbe final
    {
        std::string_view define;
        std::string_view expectedError;
    };
    constexpr std::array<BrokenProbe, 6> brokenProbes{{
        {"FAKE_SLANGC_SPIRV_PROBE=fail", "fake slangc: spirv-probe failed"},
        {"FAKE_SLANGC_METAL_PROBE=fail", "fake slangc: metal-probe failed"},
        {"FAKE_SLANGC_SPIRV_PROBE=silent", "SPIR-V reflection probe wrote no reflection JSON"},
        {"FAKE_SLANGC_METAL_PROBE=silent", "Metal reflection probe wrote no reflection JSON"},
        {"FAKE_SLANGC_SPIRV_PROBE=malformed", "SPIR-V reflection JSON parse failed"},
        {"FAKE_SLANGC_METAL_PROBE=malformed", "Metal reflection JSON parse failed"},
    }};
    for (std::size_t index = 0; index < brokenProbes.size(); ++index)
    {
        const auto& broken = brokenProbes[index];
        INFO("define=" << broken.define);
        const auto outputs = cookOutputsIn(temp.path / ("broken-" + std::to_string(index)));
        const auto result = runCookMetallib(temp.path, slangc, source, outputs, std::string(broken.define));
        INFO("stderr=" << result.stderrText);
        CHECK(result.status != 0);
        CHECK(result.stderrText.contains(broken.expectedError));
        CHECK_FALSE(result.stderrText.contains("fake slangc: metallib ran"));
        CHECK_FALSE(anyCookOutputExists(outputs));
    }
}

TEST_CASE("RachisShaderBinaryTool cook-metallib passes every include directory to each slangc run",
          "[size:small][rhi][shader][tooling]")
{
    ScopedTempDirectory const temp;
    const auto slangc = std::filesystem::path(RACHIS_FAKE_SLANGC_PATH).lexically_normal();
    REQUIRE(std::filesystem::is_regular_file(slangc));
    const auto source = temp.path / "shader.slang";
    writeBinaryFile(source, asBytes(fakeCookSource));
    const auto first = temp.path / "first include";
    const auto second = temp.path / "second";
    const auto cook = [&](std::string_view expectedIncludes, const CookOutputs& outputs) {
        return runShaderBinaryTool(temp.path,
                                   {"cook-metallib",
                                    "--slangc",
                                    slangc,
                                    "--source",
                                    source,
                                    "--entry",
                                    "computeMain",
                                    "--output",
                                    outputs.metallib,
                                    "--json",
                                    outputs.json,
                                    "--pnms",
                                    outputs.pnms,
                                    "--include-dir",
                                    first,
                                    "--define",
                                    "FAKE_SLANGC_INCLUDES=" + std::string(expectedIncludes),
                                    "--include-dir",
                                    second});
    };

    const auto cooked = cookOutputsIn(temp.path / "cooked");
    const auto result = cook("2", cooked);
    INFO("stderr=" << result.stderrText);
    REQUIRE(result.status == 0);
    CHECK(result.stderrText.contains("fake slangc: metallib includes " + first.string()));
    CHECK(result.stderrText.contains("fake slangc: metallib includes " + second.string()));
    CHECK(std::filesystem::is_regular_file(cooked.pnms));

    const auto mismatched = cookOutputsIn(temp.path / "mismatched");
    const auto rejected = cook("3", mismatched);
    INFO("stderr=" << rejected.stderrText);
    CHECK(rejected.status != 0);
    CHECK(rejected.stderrText.contains("fake slangc: spirv-probe got 2 include directories"));
    CHECK_FALSE(anyCookOutputExists(mismatched));

    const auto missing = runShaderBinaryTool(temp.path,
                                             {"cook-metallib",
                                              "--slangc",
                                              slangc,
                                              "--source",
                                              source,
                                              "--entry",
                                              "computeMain",
                                              "--output",
                                              cooked.metallib,
                                              "--json",
                                              cooked.json,
                                              "--pnms",
                                              cooked.pnms,
                                              "--include-dir"});
    CHECK(missing.status != 0);
    CHECK(missing.stderrText.contains("Missing value for argument: --include-dir"));
}
TEST_CASE("RachisShaderBinaryTool cook-metallib reports the command when slangc cannot start",
          "[size:small][rhi][shader][tooling]")
{
    ScopedTempDirectory const temp;
    const auto source = temp.path / "shader.slang";
    writeBinaryFile(source, asBytes(fakeCookSource));
    const auto missingSlangc = temp.path / "missing-slangc";
    const auto outputs = cookOutputsIn(temp.path / "cooked");

    const auto result = runCookMetallib(temp.path, missingSlangc, source, outputs, "RACHIS_METAL=1");
    INFO("stderr=" << result.stderrText);
    CHECK(result.status != 0);
    CHECK(result.stderrText.contains("Failed to start"));
    CHECK(result.stderrText.contains(missingSlangc.string()));
#ifdef _WIN32
    CHECK(result.stderrText.contains("(GetLastError "));
#else
    CHECK(result.stderrText.contains("(errno "));
#endif
    CHECK_FALSE(anyCookOutputExists(outputs));
}

TEST_CASE("RachisShaderBinaryTool reads, writes, and runs paths outside the ANSI code page",
          "[size:small][rhi][shader][tooling]")
{
    ScopedTempDirectory const temp;
    const auto directory = temp.path / std::filesystem::path(u8"rachis-üж中");
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    REQUIRE_FALSE(ec);
    const auto builtSlangc = std::filesystem::path(RACHIS_FAKE_SLANGC_PATH).lexically_normal();
    const auto slangc = directory / builtSlangc.filename();
    std::filesystem::copy_file(builtSlangc, slangc, ec);
    REQUIRE_FALSE(ec);
    std::filesystem::permissions(slangc,
                                 std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::add,
                                 ec);
    REQUIRE_FALSE(ec);
    const auto source = directory / "shader.slang";
    writeBinaryFile(source, asBytes(fakeCookSource));

    const auto outputs = cookOutputsIn(directory / "cooked");
    const auto cookResult = runCookMetallib(temp.path, slangc, source, outputs, "RACHIS_METAL=1");
    INFO("stderr=" << cookResult.stderrText);
    REQUIRE(cookResult.status == 0);
    CHECK(readTextFile(outputs.json).contains(fakeCookOutputBinding));
    REQUIRE(std::filesystem::is_regular_file(outputs.pnms));

    const auto header = directory / "Shader.h";
    const auto embedResult = runShaderBinaryTool(
        temp.path, {"embed-header", "--input", outputs.pnms, "--output", header, "--symbol", "g_rachisShader"});
    INFO("stderr=" << embedResult.stderrText);
    REQUIRE(embedResult.status == 0);
    const auto expectedHeader =
        rachis::shader_tool::generateEmbeddedBinaryHeader(readBinaryFile(outputs.pnms), "g_rachisShader");
    REQUIRE(expectedHeader.has_value());
    CHECK(readTextFile(header) == *expectedHeader);
}
#endif
#endif

TEST_CASE("ShaderBinaryTool plans per-space Metal shifts and enriches reflection JSON",
          "[size:small][rhi][shader][tooling]")
{
    constexpr std::string_view spirvJson = R"({
        "parameters": [
            {"name": "push\"block", "binding": {"kind": "descriptorTableSlot", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set0Ubo", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set0Tex", "binding": {"kind": "descriptorTableSlot", "index": 1}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set0Smp", "binding": {"kind": "descriptorTableSlot", "index": 2}, "type": {"kind": "samplerState"}},
            {"name": "set0Out", "binding": {"kind": "descriptorTableSlot", "index": 3}, "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "set1Ubo", "binding": {"kind": "descriptorTableSlot", "space": 1, "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set1Tex", "binding": {"kind": "descriptorTableSlot", "space": 1, "index": 1}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set1Smp", "binding": {"kind": "descriptorTableSlot", "space": 1, "index": 2}, "type": {"kind": "samplerState"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute"}]
    })";
    constexpr std::string_view unshiftedMetalJson = R"({
        "parameters": [
            {"name": "push\"block", "binding": {"kind": "constantBuffer", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set0Ubo", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set0Tex", "binding": {"kind": "shaderResource", "index": 1}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set0Smp", "binding": {"kind": "samplerState", "index": 2}, "type": {"kind": "samplerState"}},
            {"name": "set0Out", "binding": {"kind": "constantBuffer", "index": 3}, "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "set1Ubo", "binding": {"kind": "constantBuffer", "space": 1, "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set1Tex", "binding": {"kind": "shaderResource", "space": 1, "index": 1}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set1Smp", "binding": {"kind": "samplerState", "space": 1, "index": 2}, "type": {"kind": "samplerState"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute", "threadGroupSize": [1, 1, 1]}]
    })";

    const auto planned = rachis::shader_tool::planMetalShaderBindings(spirvJson, unshiftedMetalJson, "computeMain");
    REQUIRE(planned.has_value());
    CHECK(planned->shifts.buffer[0] == 0U);
    CHECK(planned->shifts.buffer[1] == 4U);
    CHECK(planned->shifts.buffer[4] == 5U);
    CHECK(planned->shifts.texture[0] == 0U);
    CHECK(planned->shifts.texture[1] == 1U);
    CHECK(planned->shifts.sampler[0] == 0U);
    CHECK(planned->shifts.sampler[1] == 1U);
    REQUIRE(planned->bindings.size() == 8U);
    CHECK(planned->bindings[0].pushConstant);
    CHECK(planned->bindings[0].name == "push\"block");
    CHECK_FALSE(planned->bindings[1].pushConstant);
    CHECK(planned->bindings[1].set == 0U);
    CHECK(planned->bindings[1].binding == 0U);
    CHECK(planned->bindings[5].set == 1U);
    CHECK(planned->bindings[5].binding == 0U);

    constexpr std::string_view vkPushSpirvJson = R"({
        "parameters": [
            {"name": "vkPush", "binding": {"kind": "pushConstantBuffer"}, "type": {"kind": "constantBuffer"}},
            {"name": "vkSet0", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute"}]
    })";
    constexpr std::string_view vkPushMetalJson = R"({
        "parameters": [
            {"name": "vkPush", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "vkSet0", "binding": {"kind": "constantBuffer", "index": 1}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute", "threadGroupSize": [1, 1, 1]}]
    })";
    const auto vkPlanned =
        rachis::shader_tool::planMetalShaderBindings(vkPushSpirvJson, vkPushMetalJson, "computeMain");
    REQUIRE(vkPlanned.has_value());
    CHECK(vkPlanned->shifts.buffer[0] == 0U);
    REQUIRE(vkPlanned->bindings.size() == 2U);
    CHECK(vkPlanned->bindings[0].pushConstant);
    CHECK(vkPlanned->bindings[1].set == 0U);
    CHECK(vkPlanned->bindings[1].binding == 0U);

    constexpr std::string_view shiftedMetalJson = R"({
        "parameters": [
            {"name": "push\"block", "binding": {"kind": "constantBuffer", "space": 4, "index": 5}, "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "sizes": [{"kind": "uniform", "value": 16}]}}},
            {"name": "set0Ubo", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set1Ubo", "binding": {"kind": "constantBuffer", "space": 1, "index": 4}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{
            "name": "computeMain",
            "stage": "compute",
            "threadGroupSize": [1, 1, 1],
            "bindings": [
                {"name": "push\"block", "binding": {"kind": "constantBuffer", "space": 4, "index": 5, "used": 1}},
                {"name": "set0Ubo", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}},
                {"name": "set1Ubo", "binding": {"kind": "constantBuffer", "space": 1, "index": 4, "used": 1}}
            ]
        }]
    })";
    const auto enriched = rachis::shader_tool::enrichMetalReflectionJson(shiftedMetalJson, planned->bindings);
    REQUIRE(enriched.has_value());
    CHECK(enriched->contains("\"rachisBindings\""));
    CHECK(enriched->contains("\"rachisStageInterface\": {\"inputs\": [], \"outputs\": []}"));
    CHECK_FALSE(rachis::shader_tool::enrichMetalReflectionJson(*enriched, planned->bindings).has_value());
    CHECK_FALSE(rachis::shader_tool::enrichMetalReflectionJson("[]", planned->bindings).has_value());

#if RACHIS_ENABLE_METAL
    const auto reflected =
        rachis::metal::parseSlangReflectionJson(*enriched, "computeMain", rachis::ShaderStage::compute);
    REQUIRE(reflected.has_value());
    CHECK(reflected->pushConstantBytes == 16U);
    CHECK(reflected->pushConstantMetalIndex == 5U);
    REQUIRE(reflected->resources.size() == 2U);
    CHECK(reflected->resources[0].set == 0U);
    CHECK(reflected->resources[0].binding == 0U);
    CHECK(reflected->resources[0].metalIndex == 0U);
    CHECK(reflected->resources[1].set == 1U);
    CHECK(reflected->resources[1].binding == 0U);
    CHECK(reflected->resources[1].metalIndex == 4U);
    REQUIRE(reflected->stageInterface.has_value());
    CHECK(reflected->stageInterface->inputs.empty());
    CHECK(reflected->stageInterface->outputs.empty());
#endif

    constexpr std::string_view unusedGlobalSpirvJson = R"({
        "parameters": [
            {"name": "unusedUbo", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "unusedTex", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "unusedSmp", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "samplerState"}},
            {"name": "activeOut", "binding": {"kind": "descriptorTableSlot", "index": 30}, "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "activePush", "binding": {"kind": "descriptorTableSlot", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute"}]
    })";
    constexpr std::string_view unusedGlobalMetalJson = R"({
        "parameters": [
            {"name": "unusedUbo", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "unusedTex", "binding": {"kind": "shaderResource", "index": 0}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "unusedSmp", "binding": {"kind": "samplerState", "index": 0}, "type": {"kind": "samplerState"}},
            {"name": "activeOut", "bindings": [{"kind": "constantBuffer", "index": 30}, {"kind": "unorderedAccess", "index": 0, "count": 0}], "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "activePush", "binding": {"kind": "constantBuffer", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{
            "name": "computeMain",
            "stage": "compute",
            "threadGroupSize": [1, 1, 1],
            "bindings": [
                {"name": "unusedUbo", "binding": {"kind": "constantBuffer", "index": 0, "used": 0}},
                {"name": "unusedTex", "binding": {"kind": "shaderResource", "index": 0, "used": 0}},
                {"name": "unusedSmp", "binding": {"kind": "samplerState", "index": 0, "used": 0}},
                {"name": "activeOut", "bindings": [{"kind": "constantBuffer", "index": 30, "used": 1}, {"kind": "unorderedAccess", "index": 0, "count": 0, "used": 0}]},
                {"name": "activePush", "binding": {"kind": "constantBuffer", "space": 4, "index": 0, "used": 1}}
            ]
        }]
    })";
    const auto unusedFiltered =
        rachis::shader_tool::planMetalShaderBindings(unusedGlobalSpirvJson, unusedGlobalMetalJson, "computeMain");
    REQUIRE(unusedFiltered.has_value());
    CHECK(unusedFiltered->shifts.buffer[0] == 0U);
    CHECK(unusedFiltered->shifts.buffer[4] == 0U);
    CHECK(unusedFiltered->shifts.texture[0] == 0U);
    CHECK(unusedFiltered->shifts.sampler[0] == 0U);
    REQUIRE(unusedFiltered->bindings.size() == 2U);
    CHECK(unusedFiltered->bindings[0].name == "activeOut");
    CHECK_FALSE(unusedFiltered->bindings[0].pushConstant);
    CHECK(unusedFiltered->bindings[0].set == 0U);
    CHECK(unusedFiltered->bindings[0].binding == 30U);
    CHECK(unusedFiltered->bindings[1].name == "activePush");
    CHECK(unusedFiltered->bindings[1].pushConstant);

    constexpr std::string_view sparseActiveMetalJson = R"({
        "parameters": [
            {"name": "activeUbo", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "activeOut", "bindings": [{"kind": "constantBuffer", "index": 30}], "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "activePush", "binding": {"kind": "constantBuffer", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{
            "name": "computeMain",
            "stage": "compute",
            "threadGroupSize": [1, 1, 1],
            "bindings": [
                {"name": "activeUbo", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}},
                {"name": "activeOut", "bindings": [{"kind": "constantBuffer", "index": 30, "used": 1}]},
                {"name": "activePush", "binding": {"kind": "constantBuffer", "space": 4, "index": 0, "used": 1}}
            ]
        }]
    })";
    const auto sparsePlanned = rachis::shader_tool::planMetalShaderBindings("{}", sparseActiveMetalJson, "computeMain");
    REQUIRE(sparsePlanned.has_value());
    CHECK(sparsePlanned->shifts.buffer[0] == 0U);
    CHECK(sparsePlanned->shifts.buffer[4] == 1U);
}

TEST_CASE("ShaderBinaryTool records user stage inputs and outputs from Slang Metal reflection",
          "[size:small][rhi][shader][tooling][linkage]")
{
    constexpr std::string_view vertexJson = R"({
        "entryPoints": [{
            "name": "vertexMain",
            "stage": "vertex",
            "parameters": [
                {"name": "vertexId", "semanticName": "SV_VERTEXID", "type": {"kind": "scalar", "scalarType": "uint32"}},
                {"name": "inPos", "stage": "vertex", "binding": {"kind": "varyingInput", "index": 0}, "semanticName": "POSITION",
                 "type": {"kind": "vector", "elementCount": 3, "elementType": {"kind": "scalar", "scalarType": "float32"}}},
                {"name": "extra", "stage": "vertex", "binding": {"kind": "varyingOutput", "index": 0}, "semanticName": "EXTRA",
                 "type": {"kind": "vector", "elementCount": 4, "elementType": {"kind": "scalar", "scalarType": "float32"}}}
            ],
            "result": {
                "stage": "vertex",
                "binding": {"kind": "varyingOutput", "index": 1, "count": 9},
                "type": {"kind": "struct", "name": "VSOut", "fields": [
                    {"name": "position", "semanticName": "SV_POSITION",
                     "type": {"kind": "vector", "elementCount": 4, "elementType": {"kind": "scalar", "scalarType": "float32"}}},
                    {"name": "ids", "binding": {"kind": "varyingOutput", "index": 0}, "semanticName": "TEXCOORD", "semanticIndex": 3,
                     "type": {"kind": "vector", "elementCount": 2, "elementType": {"kind": "scalar", "scalarType": "uint32"}}},
                    {"name": "inner", "binding": {"kind": "varyingOutput", "index": 1, "count": 2},
                     "type": {"kind": "struct", "name": "Inner", "fields": [
                        {"name": "a", "binding": {"kind": "varyingOutput", "index": 0}, "semanticName": "INNA",
                         "type": {"kind": "scalar", "scalarType": "float16"}},
                        {"name": "b", "binding": {"kind": "varyingOutput", "index": 1}, "semanticName": "INNB",
                         "type": {"kind": "vector", "elementCount": 3, "elementType": {"kind": "scalar", "scalarType": "int32"}}}
                     ]}},
                    {"name": "m", "binding": {"kind": "varyingOutput", "index": 3, "count": 4}, "semanticName": "MATRIX",
                     "type": {"kind": "matrix", "rowCount": 4, "columnCount": 4, "elementType": {"kind": "scalar", "scalarType": "float32"}}},
                    {"name": "m32", "binding": {"kind": "varyingOutput", "index": 7, "count": 2}, "semanticName": "MTWO",
                     "type": {"kind": "matrix", "rowCount": 3, "columnCount": 2, "elementType": {"kind": "scalar", "scalarType": "float32"}}}
                ]}
            }
        }]
    })";
    constexpr std::string_view pixelJson = R"({
        "entryPoints": [{
            "name": "pixelMain",
            "stage": "fragment",
            "parameters": [
                {"name": "pos", "semanticName": "SV_POSITION", "type": {"kind": "vector", "elementCount": 4, "elementType": {"kind": "scalar", "scalarType": "float32"}}},
                {"name": "arr", "stage": "fragment", "binding": {"kind": "varyingInput", "index": 2, "count": 2}, "semanticName": "ARR",
                 "type": {"kind": "array", "elementCount": 2, "elementType": {"kind": "vector", "elementCount": 2, "elementType": {"kind": "scalar", "scalarType": "uint16"}}}},
                {"name": "color", "stage": "fragment", "binding": {"kind": "varyingInput", "index": 0}, "semanticName": "COLOR",
                 "type": {"kind": "vector", "elementCount": 3, "elementType": {"kind": "scalar", "scalarType": "float32"}}},
                {"name": "prim", "semanticName": "sv_primitiveid", "binding": {"kind": "varyingInput", "index": 9}, "type": {"kind": "scalar", "scalarType": "uint32"}}
            ],
            "result": {"stage": "fragment", "binding": {"kind": "varyingOutput", "index": 0}, "semanticName": "SV_TARGET",
                       "type": {"kind": "vector", "elementCount": 4, "elementType": {"kind": "scalar", "scalarType": "float32"}}}
        }]
    })";

    using Entry = rachis::shader_tool::MetalStageInterfaceEntry;
    const auto vertex = rachis::shader_tool::collectMetalStageInterface(vertexJson, "vertexMain");
    REQUIRE(vertex.has_value());
    const std::vector<Entry> expectedVertexInputs{
        {.location = 0, .componentType = "float32", .componentCount = 3},
    };
    const std::vector<Entry> expectedVertexOutputs{
        {.location = 0, .componentType = "float32", .componentCount = 4},
        {.location = 1, .componentType = "uint32", .componentCount = 2},
        {.location = 2, .componentType = "float16", .componentCount = 1},
        {.location = 3, .componentType = "sint32", .componentCount = 3},
        {.location = 4, .componentType = "float32", .componentCount = 4},
        {.location = 5, .componentType = "float32", .componentCount = 4},
        {.location = 6, .componentType = "float32", .componentCount = 4},
        {.location = 7, .componentType = "float32", .componentCount = 4},
        {.location = 8, .componentType = "float32", .componentCount = 3},
        {.location = 9, .componentType = "float32", .componentCount = 3},
    };
    CHECK(vertex->inputs == expectedVertexInputs);
    CHECK(vertex->outputs == expectedVertexOutputs);

    const auto pixel = rachis::shader_tool::collectMetalStageInterface(pixelJson);
    REQUIRE(pixel.has_value());
    const std::vector<Entry> expectedPixelInputs{
        {.location = 0, .componentType = "float32", .componentCount = 3},
        {.location = 2, .componentType = "uint16", .componentCount = 2},
        {.location = 3, .componentType = "uint16", .componentCount = 2},
    };
    CHECK(pixel->inputs == expectedPixelInputs);
    CHECK(pixel->outputs.empty());

    const auto enriched = rachis::shader_tool::enrichMetalReflectionJson(pixelJson, {}, "pixelMain");
    REQUIRE(enriched.has_value());
    CHECK(enriched->contains(
        R"("rachisStageInterface": {"inputs": [{"location": 0, "component": 0, "componentType": "float32", "componentCount": 3}, )"
        R"({"location": 2, "component": 0, "componentType": "uint16", "componentCount": 2}, )"
        R"({"location": 3, "component": 0, "componentType": "uint16", "componentCount": 2}], "outputs": []})"));
    CHECK_FALSE(rachis::shader_tool::enrichMetalReflectionJson(pixelJson, {}, "otherMain").has_value());

#if RACHIS_ENABLE_METAL
    const auto reflectedPixel =
        rachis::metal::parseSlangReflectionJson(*enriched, "pixelMain", rachis::ShaderStage::pixel);
    REQUIRE(reflectedPixel.has_value());
    REQUIRE(reflectedPixel->stageInterface.has_value());
    using rachis::detail::StageComponentType;
    const std::vector<rachis::detail::StageInterfaceVariable> expectedReflectedInputs{
        {.location = 0, .componentCount = 3, .componentType = StageComponentType::float32},
        {.location = 2, .componentCount = 2, .componentType = StageComponentType::uint16},
        {.location = 3, .componentCount = 2, .componentType = StageComponentType::uint16},
    };
    CHECK(reflectedPixel->stageInterface->inputs == expectedReflectedInputs);
    CHECK(reflectedPixel->stageInterface->outputs.empty());

    constexpr std::array malformedAnnotations{
        std::string_view{
            R"({"entryPoints": [{"name": "pixelMain", "stage": "fragment"}], "rachisStageInterface": []})"},
        std::string_view{
            R"({"entryPoints": [{"name": "pixelMain", "stage": "fragment"}], "rachisStageInterface": {"inputs": []}})"},
        std::string_view{
            R"({"entryPoints": [{"name": "pixelMain", "stage": "fragment"}], "rachisStageInterface": )"
            R"({"inputs": [{"location": 0, "component": 0, "componentType": "bool", "componentCount": 1}], "outputs": []}})"},
        std::string_view{
            R"({"entryPoints": [{"name": "pixelMain", "stage": "fragment"}], "rachisStageInterface": )"
            R"({"inputs": [{"location": 0, "component": 2, "componentType": "float32", "componentCount": 3}], "outputs": []}})"},
    };
    for (const auto json : malformedAnnotations)
    {
        CAPTURE(json);
        CHECK_FALSE(rachis::metal::parseSlangReflectionJson(json, "pixelMain", rachis::ShaderStage::pixel).has_value());
    }
#endif

    constexpr std::string_view boolJson = R"({"entryPoints": [{"name": "pixelMain", "stage": "fragment", "parameters": [
        {"name": "flag", "binding": {"kind": "varyingInput", "index": 0}, "semanticName": "FLAG", "type": {"kind": "scalar", "scalarType": "bool"}}]}]})";
    const auto boolRejected = rachis::shader_tool::collectMetalStageInterface(boolJson);
    REQUIRE_FALSE(boolRejected.has_value());
    CHECK(boolRejected.error() == "Metal stage inputs and outputs of scalar type 'bool' are not supported.");

    constexpr std::string_view overlapJson =
        R"({"entryPoints": [{"name": "pixelMain", "stage": "fragment", "parameters": [
        {"name": "a", "binding": {"kind": "varyingInput", "index": 0}, "semanticName": "A", "type": {"kind": "scalar", "scalarType": "float32"}},
        {"name": "b", "binding": {"kind": "varyingInput", "index": 0}, "semanticName": "B", "type": {"kind": "scalar", "scalarType": "float32"}}]}]})";
    const auto overlapRejected = rachis::shader_tool::collectMetalStageInterface(overlapJson);
    REQUIRE_FALSE(overlapRejected.has_value());
    CHECK(overlapRejected.error() == "Two Metal stage inputs or outputs share location 0.");

    constexpr std::string_view twoEntriesJson =
        R"({"entryPoints": [{"name": "vertexMain", "stage": "vertex"}, {"name": "pixelMain", "stage": "fragment"}]})";
    CHECK_FALSE(rachis::shader_tool::collectMetalStageInterface(twoEntriesJson).has_value());
    CHECK(rachis::shader_tool::collectMetalStageInterface(twoEntriesJson, "pixelMain").has_value());
}

TEST_CASE("ShaderBinaryTool parses reflection JSON numbers under a comma-decimal C locale",
          "[size:small][rhi][shader][tooling]")
{
    ScopedCommaDecimalNumericLocale const locale;
    if (!locale.active())
    {
        SKIP("No C locale with a comma decimal separator is installed.");
    }

    constexpr std::string_view metalJson = R"({
        "parameters": [
            {"name": "weights", "binding": {"kind": "constantBuffer", "index": 2.0}, "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": [{"name": "scale", "defaultValue": 0.25}]}}},
            {"name": "blend", "binding": {"kind": "constantBuffer", "index": 1.5e1}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute", "threadGroupSize": [1, 1, 1]}]
    })";
    const auto planned = rachis::shader_tool::planMetalShaderBindings("{}", metalJson, "computeMain");
    REQUIRE(planned.has_value());
    REQUIRE(planned->bindings.size() == 2U);
    CHECK(planned->bindings[0].name == "weights");
    CHECK(planned->bindings[0].binding == 2U);
    CHECK(planned->bindings[1].name == "blend");
    CHECK(planned->bindings[1].binding == 15U);
}

TEST_CASE("ShaderBinaryTool keeps binding ranges that end past 2^32 from wrapping",
          "[size:small][rhi][shader][tooling]")
{
    constexpr std::string_view atlasSpirvJson = R"({
        "parameters": [
            {"name": "atlas", "binding": {"kind": "descriptorTableSlot", "index": 1}, "type": {"kind": "array", "elementCount": 65536, "elementType": {"kind": "array", "elementCount": 65536, "elementType": {"kind": "resource", "baseShape": "texture2D"}}}},
            {"name": "lut", "binding": {"kind": "descriptorTableSlot", "index": 2}, "type": {"kind": "resource", "baseShape": "texture2D"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute"}]
    })";
    constexpr std::string_view atlasMetalJson = R"({
        "parameters": [
            {"name": "atlas", "binding": {"kind": "shaderResource", "index": 1}, "type": {"kind": "array", "elementCount": 65536, "elementType": {"kind": "array", "elementCount": 65536, "elementType": {"kind": "resource", "baseShape": "texture2D"}}}},
            {"name": "lut", "binding": {"kind": "shaderResource", "index": 0}, "type": {"kind": "resource", "baseShape": "texture2D"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute", "threadGroupSize": [1, 1, 1]}]
    })";
    const auto atlasPlanned =
        rachis::shader_tool::planMetalShaderBindings(atlasSpirvJson, atlasMetalJson, "computeMain");
    REQUIRE(atlasPlanned.has_value());
    REQUIRE(atlasPlanned->bindings.size() == 2U);
    CHECK(atlasPlanned->bindings[0].name == "atlas");
    CHECK(atlasPlanned->bindings[0].binding == 1U);
    CHECK(atlasPlanned->bindings[1].name == "lut");
    CHECK(atlasPlanned->bindings[1].binding == 0U);

    constexpr std::string_view farIndexMetalJson = R"({
        "parameters": [
            {"name": "set0Tex", "binding": {"kind": "shaderResource", "index": 50}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set1Tex", "binding": {"kind": "shaderResource", "space": 1, "index": 0}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set1FarTex", "binding": {"kind": "shaderResource", "space": 1, "index": 4294967255}, "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "set2Tex", "binding": {"kind": "shaderResource", "space": 2, "index": 10}, "type": {"kind": "resource", "baseShape": "texture2D"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute", "threadGroupSize": [1, 1, 1]}]
    })";
    const auto farPlanned = rachis::shader_tool::planMetalShaderBindings("{}", farIndexMetalJson, "computeMain");
    REQUIRE(farPlanned.has_value());
    CHECK(farPlanned->shifts.texture[0] == 0U);
    CHECK(farPlanned->shifts.texture[1] == 51U);
    CHECK(farPlanned->shifts.texture[2] == 0U);
}

TEST_CASE("ShaderBinaryTool reads boolean Metal used flags like integer flags", "[size:small][rhi][shader][tooling]")
{
    constexpr std::string_view spirvJson = R"({
        "parameters": [
            {"name": "unusedUbo", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "activeOut", "binding": {"kind": "descriptorTableSlot", "index": 30}, "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "activePush", "binding": {"kind": "descriptorTableSlot", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{"name": "computeMain", "stage": "compute"}]
    })";
    constexpr std::string_view booleanMetalJson = R"({
        "parameters": [
            {"name": "unusedUbo", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "activeOut", "bindings": [{"kind": "constantBuffer", "index": 30}, {"kind": "unorderedAccess", "index": 0, "count": 0}], "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "activePush", "binding": {"kind": "constantBuffer", "space": 4, "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{
            "name": "computeMain",
            "stage": "compute",
            "threadGroupSize": [1, 1, 1],
            "bindings": [
                {"name": "unusedUbo", "binding": {"kind": "constantBuffer", "index": 0, "used": false}},
                {"name": "activeOut", "bindings": [{"kind": "constantBuffer", "index": 30, "used": true}, {"kind": "unorderedAccess", "index": 0, "count": 0, "used": false}]},
                {"name": "activePush", "binding": {"kind": "constantBuffer", "space": 4, "index": 0, "used": true}}
            ]
        }]
    })";

    const auto planned = rachis::shader_tool::planMetalShaderBindings(spirvJson, booleanMetalJson, "computeMain");
    REQUIRE(planned.has_value());
    REQUIRE(planned->bindings.size() == 2U);
    CHECK(planned->bindings[0].name == "activeOut");
    CHECK_FALSE(planned->bindings[0].pushConstant);
    CHECK(planned->bindings[0].set == 0U);
    CHECK(planned->bindings[0].binding == 30U);
    CHECK(planned->bindings[1].name == "activePush");
    CHECK(planned->bindings[1].pushConstant);
    CHECK(planned->shifts.buffer[0] == 0U);
    CHECK(planned->shifts.buffer[4] == 0U);
}

TEST_CASE("ShaderBinaryTool rebases the Metal vertex_id of a vertex entry point on base_vertex",
          "[size:small][rhi][shader][tooling]")
{
    constexpr std::string_view drawIdSource =
        "[[vertex]] drawIdVertex_Result_0 drawIdVertex(uint vertexId_0 [[vertex_id]], uint instanceId_0 "
        "[[instance_id]], uint base_instance_0 [[base_instance]])\n"
        "{\n"
        "\n"
        "#line 1\n"
        "    uint _S1 = instanceId_0 - base_instance_0;\n"
        "    uint triangleId_0 = vertexId_0 / 3U;\n"
        "}\n";
    const auto drawId = rachis::shader_tool::rebaseMetalVertexIds(drawIdSource);
    REQUIRE(drawId.has_value());
    REQUIRE(drawId->has_value());
    CHECK(**drawId == "[[vertex]] drawIdVertex_Result_0 drawIdVertex(uint rachis_vertex_id [[vertex_id]], uint "
                      "rachis_base_vertex [[base_vertex]], uint instanceId_0 [[instance_id]], uint base_instance_0 "
                      "[[base_instance]])\n"
                      "{\n"
                      "    uint vertexId_0 = rachis_vertex_id - rachis_base_vertex;\n"
                      "\n"
                      "#line 1\n"
                      "    uint _S1 = instanceId_0 - base_instance_0;\n"
                      "    uint triangleId_0 = vertexId_0 / 3U;\n"
                      "}\n");

    constexpr std::string_view startLocationSource =
        "[[vertex]] Out_0 startLocation(uint vertexId_0 [[vertex_id]], uint start_0 [[base_vertex]])\n"
        "{\n"
        "    uint id_0 = vertexId_0 + start_0;\n"
        "}\n";
    const auto startLocation = rachis::shader_tool::rebaseMetalVertexIds(startLocationSource);
    REQUIRE(startLocation.has_value());
    REQUIRE(startLocation->has_value());
    CHECK(**startLocation == "[[vertex]] Out_0 startLocation(uint rachis_vertex_id [[vertex_id]], uint start_0 "
                             "[[base_vertex]])\n"
                             "{\n"
                             "    uint vertexId_0 = rachis_vertex_id - start_0;\n"
                             "    uint id_0 = vertexId_0 + start_0;\n"
                             "}\n");

    constexpr std::string_view resourceSource =
        "struct Params_0 { uint rachis_vertex_id; };\n"
        "[[vertex]] Out_0 withResources(SLANG_ParameterGroup_Params_0 constant* Params_0 [[buffer(0)]], "
        "texture2d<float, access::read> image_0 [[texture(0)]], int vertexId_0 [[vertex_id]])\n"
        "{\n"
        "    return Out_0{};\n"
        "}\n";
    const auto resources = rachis::shader_tool::rebaseMetalVertexIds(resourceSource);
    REQUIRE(resources.has_value());
    REQUIRE(resources->has_value());
    CHECK(**resources == "struct Params_0 { uint rachis_vertex_id; };\n"
                         "[[vertex]] Out_0 withResources(SLANG_ParameterGroup_Params_0 constant* Params_0 "
                         "[[buffer(0)]], texture2d<float, access::read> image_0 [[texture(0)]], int "
                         "rachis_vertex_id_0 [[vertex_id]], uint rachis_base_vertex [[base_vertex]])\n"
                         "{\n"
                         "    int vertexId_0 = rachis_vertex_id_0 - rachis_base_vertex;\n"
                         "    return Out_0{};\n"
                         "}\n");
}

TEST_CASE("ShaderBinaryTool leaves Metal sources without a vertex_id parameter unchanged",
          "[size:small][rhi][shader][tooling]")
{
    constexpr std::array<std::string_view, 4> unchangedSources{{
        "// fake metal source\n",
        "[[vertex]] Out_0 noVertexId(vertexInput_0 _S1 [[stage_in]], uint instanceId_0 [[instance_id]], uint "
        "base_instance_0 [[base_instance]])\n{\n    return Out_0{};\n}\n",
        "[[fragment]] pixelOutput_0 drawIdPixel(pixelInput_0 _S1 [[stage_in]], float4 position_0 [[position]])\n"
        "{\n    return pixelOutput_0{};\n}\n",
        "[[kernel]] void fillMain(uint3 id_0 [[thread_position_in_grid]], device uint* values_0 [[buffer(0)]])\n"
        "{\n    values_0[id_0.x] = id_0.x;\n}\n",
    }};
    for (const std::string_view source : unchangedSources)
    {
        CAPTURE(source);
        const auto rebased = rachis::shader_tool::rebaseMetalVertexIds(source);
        REQUIRE(rebased.has_value());
        CHECK_FALSE(rebased->has_value());
    }

    constexpr std::array<std::string_view, 3> malformedSources{{
        "[[vertex]] Out_0 unclosed(uint vertexId_0 [[vertex_id]]\n{\n}\n",
        "[[vertex]] Out_0 unnamed([[vertex_id]])\n{\n}\n",
        "[[vertex]] Out_0 declarationOnly(uint vertexId_0 [[vertex_id]]);\n",
    }};
    for (const std::string_view source : malformedSources)
    {
        CAPTURE(source);
        CHECK_FALSE(rachis::shader_tool::rebaseMetalVertexIds(source).has_value());
    }
}
