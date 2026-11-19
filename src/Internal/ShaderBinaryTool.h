#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace rachis::shader_tool
{

    inline constexpr std::size_t metalShiftSpaceCount = 6;

    struct MetalSpaceShifts final
    {
        std::array<std::uint32_t, metalShiftSpaceCount> buffer{};
        std::array<std::uint32_t, metalShiftSpaceCount> texture{};
        std::array<std::uint32_t, metalShiftSpaceCount> sampler{};
    };

    struct MetalLogicalParameterBinding final
    {
        std::string name;
        bool pushConstant = false;
        bool drawIndex = false;
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
    };

    struct MetalCookPlan final
    {
        MetalSpaceShifts shifts{};
        std::vector<MetalLogicalParameterBinding> bindings{};
    };

    inline constexpr std::uint32_t metalArgumentBufferResourceLimit = 500'000;
    inline constexpr std::uint32_t metalArgumentBufferSamplerLimit = 1024;
    inline constexpr std::uint32_t metalArgumentTableBufferSlots = 31;

    struct MetalUnboundedArrayParameter final
    {
        std::string parameterName;
        std::string baseName;
        std::string elementType;
    };

    struct MetalArgumentBufferArray final
    {
        std::string name;
        std::string parameterName;
        std::string elementType;
        std::string_view resourceType;
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        std::uint32_t metalIndex = 0;
        std::uint32_t capacity = 0;
    };

    namespace detail
    {

        [[nodiscard]] constexpr bool isAsciiIdentifierStart(char ch) noexcept
        {
            return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
        }

        [[nodiscard]] constexpr bool isAsciiIdentifierContinue(char ch) noexcept
        {
            return isAsciiIdentifierStart(ch) || (ch >= '0' && ch <= '9');
        }

        [[nodiscard]] constexpr bool isValidCppIdentifier(std::string_view symbol) noexcept
        {
            if (symbol.empty() || !isAsciiIdentifierStart(symbol.front()))
            {
                return false;
            }
            for (const char ch : symbol.substr(1))
            {
                if (!isAsciiIdentifierContinue(ch))
                {
                    return false;
                }
            }
            return true;
        }

        constexpr void writeU32LittleEndian(std::span<std::byte, 4> destination, std::uint32_t value) noexcept
        {
            destination[0] = static_cast<std::byte>(value & 0xFFU);
            destination[1] = static_cast<std::byte>((value >> 8U) & 0xFFU);
            destination[2] = static_cast<std::byte>((value >> 16U) & 0xFFU);
            destination[3] = static_cast<std::byte>((value >> 24U) & 0xFFU);
        }

        struct JsonValue;
        using JsonArray = std::vector<JsonValue>;
        using JsonObject = std::vector<std::pair<std::string, JsonValue>>;

        struct JsonValue final
        {
            std::variant<std::monostate, bool, double, std::string, JsonArray, JsonObject> data{};

            [[nodiscard]] const JsonObject* asObject() const noexcept
            {
                return std::get_if<JsonObject>(&data);
            }
            [[nodiscard]] const JsonArray* asArray() const noexcept
            {
                return std::get_if<JsonArray>(&data);
            }
            [[nodiscard]] const std::string* asString() const noexcept
            {
                return std::get_if<std::string>(&data);
            }
            [[nodiscard]] const double* asNumber() const noexcept
            {
                return std::get_if<double>(&data);
            }
            [[nodiscard]] const bool* asBool() const noexcept
            {
                return std::get_if<bool>(&data);
            }

            [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept
            {
                const auto* object = asObject();
                if (object == nullptr)
                {
                    return nullptr;
                }
                for (const auto& [entryKey, entryValue] : *object)
                {
                    if (entryKey == key)
                    {
                        return &entryValue;
                    }
                }
                return nullptr;
            }

            [[nodiscard]] std::optional<std::uint32_t> asUint32() const noexcept
            {
                const auto* number = asNumber();
                if (number == nullptr || !std::isfinite(*number) || *number < 0.0 ||
                    *number > static_cast<double>(std::numeric_limits<std::uint32_t>::max()) ||
                    std::floor(*number) != *number)
                {
                    return std::nullopt;
                }
                return static_cast<std::uint32_t>(*number);
            }
        };

        class JsonParser final
        {
        public:
            explicit constexpr JsonParser(std::string_view text) noexcept : text_(text)
            {
            }

            [[nodiscard]] std::expected<JsonValue, std::string> parseDocument()
            {
                skipWhitespace();
                auto root = parseValue(0);
                if (!root.has_value())
                {
                    return root;
                }
                skipWhitespace();
                if (cursor_ != text_.size())
                {
                    return std::unexpected("Unexpected trailing characters in reflection JSON.");
                }
                return root;
            }

        private:
            static constexpr std::uint32_t maxDepth = 128;

            constexpr void skipWhitespace() noexcept
            {
                while (cursor_ < text_.size())
                {
                    const char ch = text_[cursor_];
                    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
                    {
                        ++cursor_;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            [[nodiscard]] static std::optional<std::uint16_t> parseHex4(std::string_view hex) noexcept
            {
                if (hex.size() != 4)
                {
                    return std::nullopt;
                }
                std::uint16_t value = 0;
                for (const char ch : hex)
                {
                    value = static_cast<std::uint16_t>(value << 4U);
                    if (ch >= '0' && ch <= '9')
                    {
                        value = static_cast<std::uint16_t>(value | static_cast<std::uint16_t>(ch - '0'));
                    }
                    else if (ch >= 'a' && ch <= 'f')
                    {
                        value = static_cast<std::uint16_t>(value | static_cast<std::uint16_t>(10 + (ch - 'a')));
                    }
                    else if (ch >= 'A' && ch <= 'F')
                    {
                        value = static_cast<std::uint16_t>(value | static_cast<std::uint16_t>(10 + (ch - 'A')));
                    }
                    else
                    {
                        return std::nullopt;
                    }
                }
                return value;
            }

            static void appendUtf8(std::string& out, std::uint32_t codePoint)
            {
                if (codePoint <= 0x7FU)
                {
                    out.push_back(static_cast<char>(codePoint));
                }
                else if (codePoint <= 0x7FFU)
                {
                    out.push_back(static_cast<char>(0xC0U | ((codePoint >> 6U) & 0x1FU)));
                    out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
                }
                else if (codePoint <= 0xFFFFU)
                {
                    out.push_back(static_cast<char>(0xE0U | ((codePoint >> 12U) & 0x0FU)));
                    out.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
                    out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
                }
                else
                {
                    out.push_back(static_cast<char>(0xF0U | ((codePoint >> 18U) & 0x07U)));
                    out.push_back(static_cast<char>(0x80U | ((codePoint >> 12U) & 0x3FU)));
                    out.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
                    out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
                }
            }

            [[nodiscard]] std::expected<std::uint32_t, std::string> parseUnicodeEscape()
            {
                if (cursor_ + 4 > text_.size())
                {
                    return std::unexpected("Truncated unicode escape in JSON string.");
                }
                const auto high = parseHex4(text_.substr(cursor_, 4));
                if (!high.has_value())
                {
                    return std::unexpected("Invalid unicode escape in JSON string.");
                }
                cursor_ += 4;
                std::uint32_t codePoint = *high;
                if (codePoint >= 0xD800U && codePoint <= 0xDBFFU)
                {
                    if (cursor_ + 6 > text_.size() || text_[cursor_] != '\\' || text_[cursor_ + 1] != 'u')
                    {
                        return std::unexpected("Missing UTF-16 low surrogate in JSON string.");
                    }
                    const auto low = parseHex4(text_.substr(cursor_ + 2, 4));
                    if (!low.has_value() || *low < 0xDC00U || *low > 0xDFFFU)
                    {
                        return std::unexpected("Invalid UTF-16 low surrogate in JSON string.");
                    }
                    cursor_ += 6;
                    codePoint = 0x10000U + (((codePoint - 0xD800U) << 10U) | (*low - 0xDC00U));
                }
                else if (codePoint >= 0xDC00U && codePoint <= 0xDFFFU)
                {
                    return std::unexpected("Unpaired UTF-16 low surrogate in JSON string.");
                }
                return codePoint;
            }

            [[nodiscard]] std::expected<std::string, std::string> parseString()
            {
                if (cursor_ >= text_.size() || text_[cursor_] != '"')
                {
                    return std::unexpected("Expected JSON string.");
                }
                ++cursor_;
                std::string result;
                while (cursor_ < text_.size())
                {
                    const char ch = text_[cursor_++];
                    if (ch == '"')
                    {
                        return result;
                    }
                    if (static_cast<unsigned char>(ch) < 0x20U)
                    {
                        return std::unexpected("Unescaped control character in JSON string.");
                    }
                    if (ch != '\\')
                    {
                        result.push_back(ch);
                        continue;
                    }
                    if (cursor_ >= text_.size())
                    {
                        return std::unexpected("Unterminated escape sequence in JSON string.");
                    }
                    const char escaped = text_[cursor_++];
                    switch (escaped)
                    {
                    case '"':
                    case '\\':
                    case '/':
                        result.push_back(escaped);
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u': {
                        auto codePoint = parseUnicodeEscape();
                        if (!codePoint.has_value())
                        {
                            return std::unexpected(std::move(codePoint.error()));
                        }
                        appendUtf8(result, *codePoint);
                        break;
                    }
                    default:
                        return std::unexpected("Unsupported escape sequence in JSON string.");
                    }
                }
                return std::unexpected("Unterminated JSON string.");
            }

            [[nodiscard]] std::expected<JsonValue, std::string> parseNumber()
            {
                const std::size_t start = cursor_;
                if (cursor_ < text_.size() && text_[cursor_] == '-')
                {
                    ++cursor_;
                }
                if (cursor_ >= text_.size())
                {
                    return std::unexpected("Invalid JSON number.");
                }
                if (text_[cursor_] == '0')
                {
                    ++cursor_;
                }
                else if (text_[cursor_] >= '1' && text_[cursor_] <= '9')
                {
                    while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9')
                    {
                        ++cursor_;
                    }
                }
                else
                {
                    return std::unexpected("Invalid JSON number.");
                }
                if (cursor_ < text_.size() && text_[cursor_] == '.')
                {
                    ++cursor_;
                    if (cursor_ >= text_.size() || text_[cursor_] < '0' || text_[cursor_] > '9')
                    {
                        return std::unexpected("Invalid fractional JSON number.");
                    }
                    while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9')
                    {
                        ++cursor_;
                    }
                }
                if (cursor_ < text_.size() && (text_[cursor_] == 'e' || text_[cursor_] == 'E'))
                {
                    ++cursor_;
                    if (cursor_ < text_.size() && (text_[cursor_] == '+' || text_[cursor_] == '-'))
                    {
                        ++cursor_;
                    }
                    if (cursor_ >= text_.size() || text_[cursor_] < '0' || text_[cursor_] > '9')
                    {
                        return std::unexpected("Invalid exponent in JSON number.");
                    }
                    while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9')
                    {
                        ++cursor_;
                    }
                }
                const std::string_view token = text_.substr(start, cursor_ - start);
                double parsed = 0.0;
                const auto [endPtr, ec] =
                    std::from_chars(token.data(), token.data() + token.size(), parsed, std::chars_format::general);
                if (ec != std::errc{} || endPtr != token.data() + token.size() || !std::isfinite(parsed))
                {
                    return std::unexpected("Out-of-range JSON number.");
                }
                return JsonValue{parsed};
            }

            [[nodiscard]] constexpr bool consume(char expected) noexcept
            {
                if (cursor_ < text_.size() && text_[cursor_] == expected)
                {
                    ++cursor_;
                    return true;
                }
                return false;
            }

            [[nodiscard]] std::expected<JsonValue, std::string> parseObjectMembers(std::uint32_t depth)
            {
                skipWhitespace();
                JsonObject object;
                if (consume('}'))
                {
                    return JsonValue{std::move(object)};
                }
                while (true)
                {
                    skipWhitespace();
                    auto key = parseString();
                    if (!key.has_value())
                    {
                        return std::unexpected(std::move(key.error()));
                    }
                    skipWhitespace();
                    if (!consume(':'))
                    {
                        return std::unexpected("Expected ':' after JSON object key.");
                    }
                    auto value = parseValue(depth + 1);
                    if (!value.has_value())
                    {
                        return std::unexpected(std::move(value.error()));
                    }
                    object.emplace_back(std::move(*key), std::move(*value));
                    skipWhitespace();
                    if (consume(','))
                    {
                        continue;
                    }
                    if (consume('}'))
                    {
                        return JsonValue{std::move(object)};
                    }
                    return std::unexpected("Expected ',' or '}' in JSON object.");
                }
            }

            [[nodiscard]] std::expected<JsonValue, std::string> parseArrayElements(std::uint32_t depth)
            {
                skipWhitespace();
                JsonArray list;
                if (consume(']'))
                {
                    return JsonValue{std::move(list)};
                }
                while (true)
                {
                    auto item = parseValue(depth + 1);
                    if (!item.has_value())
                    {
                        return std::unexpected(std::move(item.error()));
                    }
                    list.push_back(std::move(*item));
                    skipWhitespace();
                    if (consume(','))
                    {
                        continue;
                    }
                    if (consume(']'))
                    {
                        return JsonValue{std::move(list)};
                    }
                    return std::unexpected("Expected ',' or ']' in JSON array.");
                }
            }

            [[nodiscard]] std::expected<JsonValue, std::string> parseValue(std::uint32_t depth)
            {
                if (depth > maxDepth)
                {
                    return std::unexpected("Reflection JSON exceeds maximum nesting depth.");
                }
                skipWhitespace();
                if (cursor_ >= text_.size())
                {
                    return std::unexpected("Unexpected end of JSON input.");
                }
                const char ch = text_[cursor_];
                if (ch == '"')
                {
                    auto str = parseString();
                    if (!str.has_value())
                    {
                        return std::unexpected(std::move(str.error()));
                    }
                    return JsonValue{std::move(*str)};
                }
                if (consume('{'))
                {
                    return parseObjectMembers(depth);
                }
                if (consume('['))
                {
                    return parseArrayElements(depth);
                }
                if (text_.substr(cursor_).starts_with("true"))
                {
                    cursor_ += 4;
                    return JsonValue{true};
                }
                if (text_.substr(cursor_).starts_with("false"))
                {
                    cursor_ += 5;
                    return JsonValue{false};
                }
                if (text_.substr(cursor_).starts_with("null"))
                {
                    cursor_ += 4;
                    return JsonValue{std::monostate{}};
                }
                if (ch == '-' || (ch >= '0' && ch <= '9'))
                {
                    return parseNumber();
                }
                return std::unexpected("Unexpected token in JSON input.");
            }

            std::string_view text_;
            std::size_t cursor_ = 0;
        };

        [[nodiscard]] inline std::string escapeJsonString(std::string_view value)
        {
            constexpr std::array<char, 16> hexDigits{
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
            std::string escaped;
            escaped.reserve(value.size() + 2);
            escaped.push_back('"');
            for (const char ch : value)
            {
                switch (ch)
                {
                case '"':
                    escaped += "\\\"";
                    break;
                case '\\':
                    escaped += "\\\\";
                    break;
                case '\b':
                    escaped += "\\b";
                    break;
                case '\f':
                    escaped += "\\f";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    if (static_cast<unsigned char>(ch) < 0x20U)
                    {
                        const auto byte = static_cast<unsigned int>(static_cast<unsigned char>(ch));
                        escaped += "\\u00";
                        escaped.push_back(hexDigits[(byte >> 4U) & 0x0FU]);
                        escaped.push_back(hexDigits[byte & 0x0FU]);
                    }
                    else
                    {
                        escaped.push_back(ch);
                    }
                    break;
                }
            }
            escaped.push_back('"');
            return escaped;
        }

        [[nodiscard]] inline const JsonValue* findBindingOfKind(const JsonValue& parameter,
                                                                std::string_view kind) noexcept
        {
            if (const auto* single = parameter.find("binding"))
            {
                if (const auto* singleKind = single->find("kind");
                    singleKind != nullptr && singleKind->asString() != nullptr && *singleKind->asString() == kind)
                {
                    return single;
                }
            }
            if (const auto* listValue = parameter.find("bindings");
                listValue != nullptr && listValue->asArray() != nullptr)
            {
                for (const auto& item : *listValue->asArray())
                {
                    if (const auto* itemKind = item.find("kind");
                        itemKind != nullptr && itemKind->asString() != nullptr && *itemKind->asString() == kind)
                    {
                        return &item;
                    }
                }
            }
            return nullptr;
        }

        [[nodiscard]] inline bool isDrawIndexPayload(const JsonValue& type) noexcept
        {
            const auto isString = [](const JsonValue* node, std::string_view value) {
                return node != nullptr && node->asString() != nullptr && *node->asString() == value;
            };
            const auto* element = type.find("elementType");
            if (const auto* layout = type.find("elementVarLayout"); element == nullptr && layout != nullptr)
            {
                element = layout->find("type");
            }
            const auto* fields = element != nullptr ? element->find("fields") : nullptr;
            if (!isString(type.find("kind"), "constantBuffer") || element == nullptr ||
                !isString(element->find("kind"), "struct") || fields == nullptr || fields->asArray() == nullptr ||
                fields->asArray()->size() != 1)
            {
                return false;
            }
            const auto* fieldType = fields->asArray()->front().find("type");
            return fieldType != nullptr && isString(fieldType->find("kind"), "scalar") &&
                   isString(fieldType->find("scalarType"), "uint32");
        }

        struct ReflectedMetalParam final
        {
            std::string name;
            std::string_view nativeKind;
            std::size_t kindClass = 0;
            std::uint32_t metalSpace = 0;
            std::uint32_t unshiftedMetalIndex = 0;
            std::uint32_t count = 1;
            bool drawIndexPayload = false;
        };

        [[nodiscard]] inline const JsonValue* findSelectedEntryPoint(const JsonValue& root,
                                                                     std::string_view entryPoint) noexcept
        {
            const auto* entries = root.find("entryPoints");
            if (entries == nullptr || entries->asArray() == nullptr)
            {
                return nullptr;
            }
            const auto& list = *entries->asArray();
            if (entryPoint.empty())
            {
                return list.size() == 1 ? &list[0] : nullptr;
            }
            for (const auto& entry : list)
            {
                const auto* name = entry.find("name");
                if (name != nullptr && name->asString() != nullptr && *name->asString() == entryPoint)
                {
                    return &entry;
                }
            }
            return nullptr;
        }

        [[nodiscard]] inline bool isEntryPointParameterUsed(const std::vector<JsonValue>& entryBindings,
                                                            std::string_view name,
                                                            std::string_view nativeKind,
                                                            bool isGlobalParameter) noexcept
        {
            for (const auto& item : entryBindings)
            {
                const auto* itemName = item.find("name");
                if (itemName == nullptr || itemName->asString() == nullptr || *itemName->asString() != name)
                {
                    continue;
                }
                const auto* bindingNode = findBindingOfKind(item, nativeKind);
                if (bindingNode == nullptr)
                {
                    return false;
                }
                if (const auto* usedNode = bindingNode->find("used"))
                {
                    if (const auto* usedFlag = usedNode->asBool())
                    {
                        return *usedFlag;
                    }
                    const auto usedValue = usedNode->asUint32();
                    return usedValue.has_value() && *usedValue != 0U;
                }
                return true;
            }
            return !isGlobalParameter;
        }

        inline std::size_t collectParameterNodes(const JsonValue& root,
                                                 std::string_view entryPoint,
                                                 std::vector<const JsonValue*>& out)
        {
            std::size_t globalCount = 0;
            if (const auto* params = root.find("parameters"); params != nullptr && params->asArray() != nullptr)
            {
                for (const auto& param : *params->asArray())
                {
                    out.push_back(&param);
                    ++globalCount;
                }
            }
            if (const auto* entry = findSelectedEntryPoint(root, entryPoint))
            {
                if (const auto* params = entry->find("parameters"); params != nullptr && params->asArray() != nullptr)
                {
                    for (const auto& param : *params->asArray())
                    {
                        out.push_back(&param);
                    }
                }
                if (const auto* scope = entry->find("scope"); scope != nullptr)
                {
                    if (const auto* scopeParams = scope->find("parameters");
                        scopeParams != nullptr && scopeParams->asArray() != nullptr)
                    {
                        for (const auto& param : *scopeParams->asArray())
                        {
                            out.push_back(&param);
                        }
                    }
                }
            }
            return globalCount;
        }

        [[nodiscard]] inline std::optional<std::uint32_t> findUint32(const JsonValue& object,
                                                                     std::string_view key) noexcept
        {
            const auto* node = object.find(key);
            if (node == nullptr)
            {
                return std::nullopt;
            }
            return node->asUint32();
        }

        struct FlattenedArrayType final
        {
            const JsonValue* elementType = nullptr;
            std::uint32_t elementCount = 1;
        };

        [[nodiscard]] inline FlattenedArrayType flattenArrayType(const JsonValue& parameterType) noexcept
        {
            const JsonValue* typeNode = &parameterType;
            std::uint32_t count = 1;
            while (typeNode != nullptr)
            {
                const auto* kindNode = typeNode->find("kind");
                if (kindNode == nullptr || kindNode->asString() == nullptr || *kindNode->asString() != "array")
                {
                    break;
                }
                if (const auto elements = findUint32(*typeNode, "elementCount"); elements.has_value() && *elements > 0)
                {
                    count = (*elements > (std::numeric_limits<std::uint32_t>::max() / count))
                                ? std::numeric_limits<std::uint32_t>::max()
                                : count * *elements;
                }
                typeNode = typeNode->find("elementType");
            }
            return FlattenedArrayType{.elementType = typeNode, .elementCount = count};
        }

        inline constexpr std::size_t metalBufferKindClass = 0;
        inline constexpr std::size_t metalTextureKindClass = 1;
        inline constexpr std::size_t metalSamplerKindClass = 2;
        inline constexpr std::size_t metalKindClassCount = 3;

        struct MetalBindingKind final
        {
            std::string_view nativeKind;
            std::size_t kindClass = 0;
        };

        [[nodiscard]] inline std::optional<MetalBindingKind> classifyMetalBindingKind(
            const JsonValue& elementType) noexcept
        {
            const auto* kindNode = elementType.find("kind");
            if (kindNode == nullptr || kindNode->asString() == nullptr)
            {
                return std::nullopt;
            }
            const std::string_view typeKind = *kindNode->asString();
            if (typeKind == "constantBuffer")
            {
                return MetalBindingKind{.nativeKind = "constantBuffer", .kindClass = metalBufferKindClass};
            }
            if (typeKind == "samplerState")
            {
                return MetalBindingKind{.nativeKind = "samplerState", .kindClass = metalSamplerKindClass};
            }
            if (typeKind != "resource")
            {
                return std::nullopt;
            }
            const auto* shapeNode = elementType.find("baseShape");
            const std::string_view baseShape = (shapeNode != nullptr && shapeNode->asString() != nullptr)
                                                   ? std::string_view{*shapeNode->asString()}
                                                   : "";
            if (baseShape.starts_with("texture"))
            {
                return MetalBindingKind{.nativeKind = "shaderResource", .kindClass = metalTextureKindClass};
            }
            if (baseShape == "structuredBuffer" || baseShape == "byteAddressBuffer" ||
                baseShape == "accelerationStructure")
            {
                return MetalBindingKind{.nativeKind = "constantBuffer", .kindClass = metalBufferKindClass};
            }
            return std::nullopt;
        }

        struct UnshiftedMetalSlot final
        {
            std::uint32_t space = 0;
            std::uint32_t index = 0;
        };

        [[nodiscard]] inline std::optional<UnshiftedMetalSlot> readUnshiftedMetalSlot(
            const JsonValue& bindingNode) noexcept
        {
            const auto unshiftedIndex = findUint32(bindingNode, "index");
            if (!unshiftedIndex.has_value())
            {
                return std::nullopt;
            }
            std::uint32_t space = 0;
            if (const auto* spaceNode = bindingNode.find("space"))
            {
                const auto parsedSpace = spaceNode->asUint32();
                if (!parsedSpace.has_value())
                {
                    return std::nullopt;
                }
                space = *parsedSpace;
            }
            return UnshiftedMetalSlot{.space = space, .index = *unshiftedIndex};
        }

        [[nodiscard]] inline std::optional<ReflectedMetalParam> classifyMetalParameter(const JsonValue& parameter)
        {
            const auto* nameNode = parameter.find("name");
            const auto* typeNode = parameter.find("type");
            if (nameNode == nullptr || nameNode->asString() == nullptr || typeNode == nullptr)
            {
                return std::nullopt;
            }
            const auto flattenedType = flattenArrayType(*typeNode);
            if (flattenedType.elementType == nullptr)
            {
                return std::nullopt;
            }
            const auto kind = classifyMetalBindingKind(*flattenedType.elementType);
            if (!kind.has_value())
            {
                return std::nullopt;
            }
            const auto* bindingNode = findBindingOfKind(parameter, kind->nativeKind);
            if (bindingNode == nullptr)
            {
                return std::nullopt;
            }
            const auto slot = readUnshiftedMetalSlot(*bindingNode);
            if (!slot.has_value())
            {
                return std::nullopt;
            }

            return ReflectedMetalParam{
                .name = *nameNode->asString(),
                .nativeKind = kind->nativeKind,
                .kindClass = kind->kindClass,
                .metalSpace = slot->space,
                .unshiftedMetalIndex = slot->index,
                .count = flattenedType.elementCount,
                .drawIndexPayload = flattenedType.elementType == typeNode && isDrawIndexPayload(*typeNode),
            };
        }

        struct ReflectionRoots final
        {
            JsonValue spirv;
            JsonValue metal;
        };

        [[nodiscard]] inline std::expected<ReflectionRoots, std::string> parseReflectionRoots(
            std::string_view spirvReflectionJson, std::string_view unshiftedMetalReflectionJson)
        {
            auto spirvRoot = JsonParser(spirvReflectionJson).parseDocument();
            if (!spirvRoot.has_value())
            {
                return std::unexpected("SPIR-V reflection JSON parse failed: " + spirvRoot.error());
            }
            auto metalRoot = JsonParser(unshiftedMetalReflectionJson).parseDocument();
            if (!metalRoot.has_value())
            {
                return std::unexpected("Metal reflection JSON parse failed: " + metalRoot.error());
            }
            if (spirvRoot->asObject() == nullptr || metalRoot->asObject() == nullptr)
            {
                return std::unexpected("Reflection JSON root must be an object.");
            }
            return ReflectionRoots{.spirv = std::move(*spirvRoot), .metal = std::move(*metalRoot)};
        }

        [[nodiscard]] inline std::expected<const JsonArray*, std::string> findSelectedEntryPointBindings(
            const JsonValue& metalRoot, std::string_view entryPoint)
        {
            const auto* selectedMetalEntry = findSelectedEntryPoint(metalRoot, entryPoint);
            const auto* entries = metalRoot.find("entryPoints");
            const bool hasEntryPointList = entries != nullptr && entries->asArray() != nullptr;
            if (!entryPoint.empty() && hasEntryPointList && selectedMetalEntry == nullptr)
            {
                return std::unexpected("Metal reflection does not contain the requested entry point.");
            }
            if (selectedMetalEntry == nullptr)
            {
                return nullptr;
            }
            const auto* bindingsNode = selectedMetalEntry->find("bindings");
            if (bindingsNode == nullptr)
            {
                return nullptr;
            }
            const auto* entryBindings = bindingsNode->asArray();
            if (entryBindings == nullptr)
            {
                return std::unexpected("Metal entry point bindings must be an array.");
            }
            return entryBindings;
        }

        [[nodiscard]] inline std::vector<ReflectedMetalParam> collectUsedMetalParameters(const JsonValue& metalRoot,
                                                                                         std::string_view entryPoint,
                                                                                         const JsonArray* entryBindings)
        {
            std::vector<const JsonValue*> parameterNodes;
            const std::size_t globalParameterCount = collectParameterNodes(metalRoot, entryPoint, parameterNodes);
            std::vector<ReflectedMetalParam> metalParams;
            metalParams.reserve(parameterNodes.size());
            for (std::size_t paramIndex = 0; paramIndex < parameterNodes.size(); ++paramIndex)
            {
                auto classified = classifyMetalParameter(*parameterNodes[paramIndex]);
                if (!classified.has_value())
                {
                    continue;
                }
                if (entryBindings != nullptr &&
                    !isEntryPointParameterUsed(
                        *entryBindings, classified->name, classified->nativeKind, paramIndex < globalParameterCount))
                {
                    continue;
                }
                if (std::ranges::any_of(metalParams,
                                        [&](const auto& existing) { return existing.name == classified->name; }))
                {
                    continue;
                }
                metalParams.push_back(std::move(*classified));
            }
            return metalParams;
        }

        struct MetalSpaceIndexRange final
        {
            bool hasAny = false;
            std::uint32_t minIndex = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t maxEnd = 0;
        };

        using MetalSpaceIndexRanges =
            std::array<std::array<MetalSpaceIndexRange, metalShiftSpaceCount>, metalKindClassCount>;

        [[nodiscard]] inline MetalSpaceIndexRanges measureSpaceIndexRanges(
            const std::vector<ReflectedMetalParam>& metalParams) noexcept
        {
            MetalSpaceIndexRanges ranges{};
            for (const auto& param : metalParams)
            {
                if (param.metalSpace >= metalShiftSpaceCount)
                {
                    continue;
                }
                auto& range = ranges[param.kindClass][param.metalSpace];
                range.hasAny = true;
                range.minIndex = std::min(range.minIndex, param.unshiftedMetalIndex);
                const std::uint32_t end =
                    param.count > (std::numeric_limits<std::uint32_t>::max() - param.unshiftedMetalIndex)
                        ? std::numeric_limits<std::uint32_t>::max()
                        : param.unshiftedMetalIndex + param.count;
                range.maxEnd = std::max(range.maxEnd, end);
            }
            return ranges;
        }

        struct MetalSlotAllocation final
        {
            std::size_t kindClass = 0;
            std::uint32_t slotLimit = 0;
            std::uint32_t nextFreeSlot = 0;
            std::vector<std::pair<std::uint64_t, std::uint64_t>> placedIntervals{};
        };

        [[nodiscard]] inline bool shiftedParameterFits(const ReflectedMetalParam& param,
                                                       std::uint32_t candidateShift,
                                                       const MetalSlotAllocation& allocation) noexcept
        {
            const std::uint64_t start = std::uint64_t{param.unshiftedMetalIndex} + candidateShift;
            const std::uint64_t end = start + param.count;
            if (end > allocation.slotLimit)
            {
                return false;
            }
            for (const auto& [placedStart, placedEnd] : allocation.placedIntervals)
            {
                if (start < placedEnd && placedStart < end)
                {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] inline bool spaceFitsAtShift(const std::vector<ReflectedMetalParam>& metalParams,
                                                   const MetalSlotAllocation& allocation,
                                                   std::size_t space,
                                                   std::uint32_t candidateShift) noexcept
        {
            for (const auto& param : metalParams)
            {
                if (param.kindClass != allocation.kindClass || param.metalSpace != space)
                {
                    continue;
                }
                if (!shiftedParameterFits(param, candidateShift, allocation))
                {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] inline std::uint32_t chooseSpaceShift(const std::vector<ReflectedMetalParam>& metalParams,
                                                            const MetalSlotAllocation& allocation,
                                                            std::size_t space,
                                                            const MetalSpaceIndexRange& range) noexcept
        {
            const std::uint32_t appendingShift =
                range.minIndex < allocation.nextFreeSlot ? (allocation.nextFreeSlot - range.minIndex) : 0U;
            if (range.maxEnd <= allocation.slotLimit - std::min(appendingShift, allocation.slotLimit))
            {
                return appendingShift;
            }
            for (std::uint32_t candidate = 0; candidate < allocation.slotLimit; ++candidate)
            {
                if (spaceFitsAtShift(metalParams, allocation, space, candidate))
                {
                    return candidate;
                }
            }
            return appendingShift;
        }

        [[nodiscard]] inline std::array<std::uint32_t, metalShiftSpaceCount> assignShiftsForKindClass(
            const std::vector<ReflectedMetalParam>& metalParams,
            const MetalSpaceIndexRanges& ranges,
            std::size_t kindClass,
            std::span<const std::size_t, metalShiftSpaceCount> spaceOrder)
        {
            constexpr std::array<std::uint32_t, metalKindClassCount> metalSlotLimits{31U, 128U, 16U};
            MetalSlotAllocation allocation{.kindClass = kindClass, .slotLimit = metalSlotLimits[kindClass]};
            allocation.placedIntervals.reserve(metalParams.size());
            std::array<std::uint32_t, metalShiftSpaceCount> shiftTable{};

            for (const std::size_t space : spaceOrder)
            {
                const auto& range = ranges[kindClass][space];
                if (!range.hasAny)
                {
                    continue;
                }
                const std::uint32_t shift = chooseSpaceShift(metalParams, allocation, space, range);
                shiftTable[space] = shift;
                if (range.maxEnd <= std::numeric_limits<std::uint32_t>::max() - shift)
                {
                    allocation.nextFreeSlot = std::max(allocation.nextFreeSlot, shift + range.maxEnd);
                }
                else
                {
                    allocation.nextFreeSlot = std::numeric_limits<std::uint32_t>::max();
                }
                for (const auto& param : metalParams)
                {
                    if (param.kindClass == kindClass && param.metalSpace == space)
                    {
                        const std::uint64_t start = std::uint64_t{param.unshiftedMetalIndex} + shift;
                        allocation.placedIntervals.emplace_back(start, start + param.count);
                    }
                }
            }
            return shiftTable;
        }

        inline constexpr std::uint32_t metalPushConstantSpace = 4;

        [[nodiscard]] inline MetalSpaceShifts planMetalSpaceShifts(const std::vector<ReflectedMetalParam>& metalParams)
        {
            constexpr std::array<std::size_t, metalShiftSpaceCount> defaultSpaceOrder{0, 1, 2, 3, 4, 5};
            constexpr std::array<std::size_t, metalShiftSpaceCount> pushFirstSpaceOrder{4, 0, 1, 2, 3, 5};
            const auto ranges = measureSpaceIndexRanges(metalParams);
            const auto& firstSpaceBuffers = ranges[metalBufferKindClass][0];
            const auto& pushSpaceBuffers = ranges[metalBufferKindClass][metalPushConstantSpace];
            const bool pushSpaceBuffersComeFirst =
                pushSpaceBuffers.hasAny &&
                (!firstSpaceBuffers.hasAny || firstSpaceBuffers.minIndex >= pushSpaceBuffers.maxEnd);
            const auto& bufferOrder = pushSpaceBuffersComeFirst ? pushFirstSpaceOrder : defaultSpaceOrder;

            MetalSpaceShifts shifts{};
            shifts.buffer = assignShiftsForKindClass(metalParams, ranges, metalBufferKindClass, bufferOrder);
            shifts.texture = assignShiftsForKindClass(metalParams, ranges, metalTextureKindClass, defaultSpaceOrder);
            shifts.sampler = assignShiftsForKindClass(metalParams, ranges, metalSamplerKindClass, defaultSpaceOrder);
            return shifts;
        }

        inline constexpr std::uint32_t metalDrawIndexSpace = 5;

        struct LogicalBindingCandidate final
        {
            ReflectedMetalParam metal;
            bool pushConstant = false;
            bool drawIndex = false;
            std::uint32_t logicalSet = 0;
            std::uint32_t logicalBinding = 0;
        };

        [[nodiscard]] inline const JsonValue* findParameterNamed(const std::vector<const JsonValue*>& parameters,
                                                                 std::string_view name) noexcept
        {
            for (const auto* parameter : parameters)
            {
                const auto* nameNode = parameter->find("name");
                if (nameNode != nullptr && nameNode->asString() != nullptr && *nameNode->asString() == name)
                {
                    return parameter;
                }
            }
            return nullptr;
        }

        [[nodiscard]] inline std::expected<LogicalBindingCandidate, std::string> resolveLogicalBinding(
            const ReflectedMetalParam& metalParam, const JsonValue* spirvParam)
        {
            LogicalBindingCandidate candidate{.metal = metalParam};
            if (metalParam.metalSpace == metalDrawIndexSpace)
            {
                if (metalParam.kindClass != metalBufferKindClass || !metalParam.drawIndexPayload ||
                    metalParam.unshiftedMetalIndex != 0 || metalParam.count != 1)
                {
                    return std::unexpected("The resource '" + metalParam.name +
                                           "' uses register space 5, which is reserved for the draw index. Only the "
                                           "constant buffer of Rachis/DrawIndex.hlsli, one uint at "
                                           "register(b0, space5), may use it.");
                }
                candidate.drawIndex = true;
                return candidate;
            }
            if (metalParam.metalSpace == metalPushConstantSpace ||
                (spirvParam != nullptr && findBindingOfKind(*spirvParam, "pushConstantBuffer") != nullptr))
            {
                candidate.pushConstant = true;
                return candidate;
            }

            candidate.logicalSet = metalParam.metalSpace;
            candidate.logicalBinding = metalParam.unshiftedMetalIndex;
            const auto* slot = spirvParam != nullptr ? findBindingOfKind(*spirvParam, "descriptorTableSlot") : nullptr;
            if (slot == nullptr)
            {
                return candidate;
            }
            if (const auto slotIndex = findUint32(*slot, "index"))
            {
                candidate.logicalBinding = *slotIndex;
            }
            candidate.logicalSet = findUint32(*slot, "space").value_or(0U);
            return candidate;
        }

        [[nodiscard]] inline std::expected<std::vector<LogicalBindingCandidate>, std::string> resolveLogicalBindings(
            const std::vector<ReflectedMetalParam>& metalParams,
            const JsonValue& spirvRoot,
            std::string_view entryPoint)
        {
            std::vector<const JsonValue*> spirvParams;
            collectParameterNodes(spirvRoot, entryPoint, spirvParams);
            std::vector<LogicalBindingCandidate> candidates;
            candidates.reserve(metalParams.size());
            for (const auto& metalParam : metalParams)
            {
                auto candidate = resolveLogicalBinding(metalParam, findParameterNamed(spirvParams, metalParam.name));
                if (!candidate)
                {
                    return std::unexpected(std::move(candidate.error()));
                }
                candidates.push_back(std::move(*candidate));
            }
            return candidates;
        }

        [[nodiscard]] inline bool isDescriptorInSameSetAndKindClass(const LogicalBindingCandidate& peer,
                                                                    const LogicalBindingCandidate& reference) noexcept
        {
            return !peer.pushConstant && !peer.drawIndex && peer.logicalSet == reference.logicalSet &&
                   peer.metal.kindClass == reference.metal.kindClass;
        }

        [[nodiscard]] inline bool bindingRangesOverlap(std::uint32_t firstStart,
                                                       std::uint32_t firstCount,
                                                       std::uint32_t secondStart,
                                                       std::uint32_t secondCount) noexcept
        {
            const std::uint64_t firstEnd = std::uint64_t{firstStart} + firstCount;
            const std::uint64_t secondEnd = std::uint64_t{secondStart} + secondCount;
            return firstStart < secondEnd && secondStart < firstEnd;
        }

        [[nodiscard]] inline bool logicalRangesOverlap(const LogicalBindingCandidate& first,
                                                       const LogicalBindingCandidate& second) noexcept
        {
            return bindingRangesOverlap(
                first.logicalBinding, first.metal.count, second.logicalBinding, second.metal.count);
        }

        [[nodiscard]] inline bool metalRangesOverlap(const LogicalBindingCandidate& first,
                                                     const LogicalBindingCandidate& second) noexcept
        {
            return bindingRangesOverlap(first.metal.unshiftedMetalIndex,
                                        first.metal.count,
                                        second.metal.unshiftedMetalIndex,
                                        second.metal.count);
        }

        [[nodiscard]] inline bool occupyDistinctMetalSlots(const LogicalBindingCandidate& first,
                                                           const LogicalBindingCandidate& second) noexcept
        {
            return first.metal.metalSpace == first.logicalSet && second.metal.metalSpace == second.logicalSet &&
                   !metalRangesOverlap(first, second);
        }

        [[nodiscard]] inline bool collidesOnlyInLogicalBindings(const std::vector<LogicalBindingCandidate>& candidates,
                                                                std::size_t index) noexcept
        {
            const auto& candidate = candidates[index];
            bool collides = false;
            for (std::size_t otherIndex = 0; otherIndex < candidates.size(); ++otherIndex)
            {
                const auto& other = candidates[otherIndex];
                if (otherIndex == index || !isDescriptorInSameSetAndKindClass(other, candidate) ||
                    !logicalRangesOverlap(candidate, other))
                {
                    continue;
                }
                if (!occupyDistinctMetalSlots(candidate, other))
                {
                    return false;
                }
                collides = true;
            }
            return collides;
        }

        inline void restoreMetalIndicesForCollidingLogicalBindings(
            std::vector<LogicalBindingCandidate>& candidates) noexcept
        {
            for (std::size_t index = 0; index < candidates.size(); ++index)
            {
                if (candidates[index].pushConstant || candidates[index].drawIndex ||
                    !collidesOnlyInLogicalBindings(candidates, index))
                {
                    continue;
                }
                for (auto& item : candidates)
                {
                    if (isDescriptorInSameSetAndKindClass(item, candidates[index]))
                    {
                        item.logicalBinding = item.metal.unshiftedMetalIndex;
                    }
                }
            }
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<MetalCookPlan, std::string> planMetalShaderBindings(
        std::string_view spirvReflectionJson,
        std::string_view unshiftedMetalReflectionJson,
        std::string_view entryPoint = {})
    {
        const auto roots = detail::parseReflectionRoots(spirvReflectionJson, unshiftedMetalReflectionJson);
        if (!roots.has_value())
        {
            return std::unexpected(roots.error());
        }
        const auto entryBindings = detail::findSelectedEntryPointBindings(roots->metal, entryPoint);
        if (!entryBindings.has_value())
        {
            return std::unexpected(entryBindings.error());
        }
        const auto metalParams = detail::collectUsedMetalParameters(roots->metal, entryPoint, *entryBindings);

        MetalCookPlan plan{};
        plan.shifts = detail::planMetalSpaceShifts(metalParams);

        auto resolved = detail::resolveLogicalBindings(metalParams, roots->spirv, entryPoint);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        auto& candidates = *resolved;
        detail::restoreMetalIndicesForCollidingLogicalBindings(candidates);
        plan.bindings.reserve(candidates.size());
        for (const auto& item : candidates)
        {
            plan.bindings.push_back(MetalLogicalParameterBinding{
                .name = item.metal.name,
                .pushConstant = item.pushConstant,
                .drawIndex = item.drawIndex,
                .set = item.logicalSet,
                .binding = item.logicalBinding,
            });
        }
        return plan;
    }

    struct MetalStageInterfaceEntry final
    {
        std::uint32_t location = 0;
        std::uint32_t component = 0;
        std::string_view componentType{};
        std::uint32_t componentCount = 0;

        friend bool operator==(const MetalStageInterfaceEntry&, const MetalStageInterfaceEntry&) = default;
    };

    struct MetalStageInterface final
    {
        std::vector<MetalStageInterfaceEntry> inputs{};
        std::vector<MetalStageInterfaceEntry> outputs{};
        bool reflected = true;
    };

    namespace detail
    {

        inline constexpr std::uint32_t stageVaryingComponentsPerLocation = 4;
        inline constexpr std::uint32_t maxStageVaryingDepth = 32;
        inline constexpr std::uint64_t maxStageVaryingSlots = std::uint64_t{stageVaryingComponentsPerLocation} << 16U;

        [[nodiscard]] inline const std::string* findString(const JsonValue& object, std::string_view key) noexcept
        {
            const auto* node = object.find(key);
            return node != nullptr ? node->asString() : nullptr;
        }

        [[nodiscard]] inline bool isSystemValueVarying(const JsonValue& node) noexcept
        {
            const auto* semantic = findString(node, "semanticName");
            if (semantic == nullptr || semantic->size() < 3)
            {
                return false;
            }
            const char first = (*semantic)[0];
            const char second = (*semantic)[1];
            return (first == 'S' || first == 's') && (second == 'V' || second == 'v') && (*semantic)[2] == '_';
        }

        struct StageScalarType final
        {
            std::string_view name;
            std::uint32_t slots = 1;
        };

        [[nodiscard]] inline std::optional<StageScalarType> stageScalarType(std::string_view slangScalarType) noexcept
        {
            constexpr std::array<std::pair<std::string_view, StageScalarType>, 9> table{{
                {"float16", {.name = "float16", .slots = 1}},
                {"float32", {.name = "float32", .slots = 1}},
                {"float64", {.name = "float64", .slots = 2}},
                {"int16", {.name = "sint16", .slots = 1}},
                {"int32", {.name = "sint32", .slots = 1}},
                {"int64", {.name = "sint64", .slots = 2}},
                {"uint16", {.name = "uint16", .slots = 1}},
                {"uint32", {.name = "uint32", .slots = 1}},
                {"uint64", {.name = "uint64", .slots = 2}},
            }};
            for (const auto& [slangName, type] : table)
            {
                if (slangName == slangScalarType)
                {
                    return type;
                }
            }
            return std::nullopt;
        }

        struct StageVaryingShape final
        {
            std::string_view componentType;
            std::uint64_t slots = 0;
        };

        [[nodiscard]] inline std::expected<StageVaryingShape, std::string> measureStageVaryingType(
            const JsonValue& type, std::uint32_t depth)
        {
            const auto* kind = findString(type, "kind");
            if (depth > maxStageVaryingDepth || kind == nullptr)
            {
                return std::unexpected("A Metal stage input or output has no readable type.");
            }
            if (*kind == "array")
            {
                const auto elements = findUint32(type, "elementCount");
                const auto* elementType = type.find("elementType");
                if (!elements || *elements == 0 || elementType == nullptr)
                {
                    return std::unexpected("A Metal stage input or output array has no element count or type.");
                }
                auto element = measureStageVaryingType(*elementType, depth + 1);
                if (!element.has_value())
                {
                    return element;
                }
                if (element->slots > maxStageVaryingSlots / *elements)
                {
                    return std::unexpected("A Metal stage input or output array is too large.");
                }
                element->slots *= *elements;
                return element;
            }

            std::uint64_t elements = 1;
            const JsonValue* scalarNode = &type;
            if (*kind == "vector")
            {
                const auto count = findUint32(type, "elementCount");
                elements = count.value_or(0);
                scalarNode = type.find("elementType");
            }
            else if (*kind == "matrix")
            {
                const auto rows = findUint32(type, "rowCount");
                const auto columns = findUint32(type, "columnCount");
                elements = (rows && columns && *rows <= 4 && *columns <= 4) ? std::uint64_t{*rows} * *columns : 0;
                scalarNode = type.find("elementType");
            }
            else if (*kind != "scalar")
            {
                return std::unexpected("A Metal stage input or output has the unsupported type kind '" + *kind + "'.");
            }
            const auto* scalarName = scalarNode != nullptr ? findString(*scalarNode, "scalarType") : nullptr;
            if (elements == 0 || scalarName == nullptr)
            {
                return std::unexpected("A Metal stage input or output has no element count or scalar type.");
            }
            const auto scalar = stageScalarType(*scalarName);
            if (!scalar.has_value())
            {
                return std::unexpected("Metal stage inputs and outputs of scalar type '" + *scalarName +
                                       "' are not supported.");
            }
            return StageVaryingShape{.componentType = scalar->name, .slots = elements * scalar->slots};
        }

        [[nodiscard]] inline std::expected<void, std::string> appendStageVaryings(
            const JsonValue& node,
            std::string_view bindingKind,
            std::uint32_t baseLocation,
            std::vector<MetalStageInterfaceEntry>& out,
            std::uint32_t depth);

        [[nodiscard]] inline std::expected<void, std::string> appendStageVaryingFields(
            const JsonValue& structType,
            std::string_view bindingKind,
            std::uint32_t baseLocation,
            std::vector<MetalStageInterfaceEntry>& out,
            std::uint32_t depth)
        {
            const auto* fields = structType.find("fields");
            if (fields == nullptr || fields->asArray() == nullptr)
            {
                return std::unexpected("A Metal stage input or output struct has no fields.");
            }
            for (const auto& field : *fields->asArray())
            {
                if (auto appended = appendStageVaryings(field, bindingKind, baseLocation, out, depth + 1);
                    !appended.has_value())
                {
                    return appended;
                }
            }
            return {};
        }

        [[nodiscard]] inline std::expected<void, std::string> appendStageVaryings(
            const JsonValue& node,
            std::string_view bindingKind,
            std::uint32_t baseLocation,
            std::vector<MetalStageInterfaceEntry>& out,
            std::uint32_t depth)
        {
            if (depth > maxStageVaryingDepth)
            {
                return std::unexpected("Metal stage inputs and outputs nest too deeply.");
            }
            if (isSystemValueVarying(node))
            {
                return {};
            }
            const auto* binding = findBindingOfKind(node, bindingKind);
            if (binding == nullptr)
            {
                return {};
            }
            const auto index = findUint32(*binding, "index");
            const auto count =
                binding->find("count") != nullptr ? findUint32(*binding, "count") : std::optional<std::uint32_t>{1};
            constexpr auto maxLocation = std::numeric_limits<std::uint32_t>::max();
            if (!index || !count || *count == 0 || *index > maxLocation - baseLocation ||
                *count > maxLocation - baseLocation - *index)
            {
                return std::unexpected("A Metal stage input or output has an invalid location.");
            }
            const std::uint32_t location = baseLocation + *index;
            const auto* type = node.find("type");
            if (type == nullptr)
            {
                return std::unexpected("A Metal stage input or output has no type.");
            }

            if (const auto* kind = findString(*type, "kind"); kind != nullptr && *kind == "struct")
            {
                return appendStageVaryingFields(*type, bindingKind, location, out, depth);
            }

            const auto shape = measureStageVaryingType(*type, depth);
            if (!shape.has_value())
            {
                return std::unexpected(shape.error());
            }
            if (shape->slots % *count != 0 || shape->slots / *count == 0 ||
                shape->slots / *count > stageVaryingComponentsPerLocation)
            {
                return std::unexpected("A Metal stage input or output does not split evenly into its locations.");
            }
            const auto componentsPerLocation = static_cast<std::uint32_t>(shape->slots / *count);
            for (std::uint32_t offset = 0; offset < *count; ++offset)
            {
                out.push_back(MetalStageInterfaceEntry{
                    .location = location + offset,
                    .component = 0,
                    .componentType = shape->componentType,
                    .componentCount = componentsPerLocation,
                });
            }
            return {};
        }

        [[nodiscard]] inline std::expected<void, std::string> sortStageInterfaceEntries(
            std::vector<MetalStageInterfaceEntry>& entries)
        {
            std::ranges::sort(entries, [](const auto& left, const auto& right) {
                return left.location != right.location ? left.location < right.location
                                                       : left.component < right.component;
            });
            for (std::size_t index = 1; index < entries.size(); ++index)
            {
                if (entries[index].location == entries[index - 1].location)
                {
                    return std::unexpected("Two Metal stage inputs or outputs share location " +
                                           std::to_string(entries[index].location) + ".");
                }
            }
            return {};
        }

        [[nodiscard]] inline std::expected<MetalStageInterface, std::string> collectMetalStageInterface(
            const JsonValue& root, std::string_view entryPoint)
        {
            const auto* entry = findSelectedEntryPoint(root, entryPoint);
            if (entry == nullptr)
            {
                return std::unexpected("Cooked Metal reflection JSON does not contain the selected entry point.");
            }
            MetalStageInterface stageInterface;
            if (const auto* stage = findString(*entry, "stage");
                stage != nullptr && (*stage == "mesh" || *stage == "amplification"))
            {
                stageInterface.reflected = false;
                return stageInterface;
            }
            if (const auto* parameters = entry->find("parameters");
                parameters != nullptr && parameters->asArray() != nullptr)
            {
                for (const auto& parameter : *parameters->asArray())
                {
                    if (auto input = appendStageVaryings(parameter, "varyingInput", 0, stageInterface.inputs, 0);
                        !input.has_value())
                    {
                        return std::unexpected(input.error());
                    }
                    if (auto output = appendStageVaryings(parameter, "varyingOutput", 0, stageInterface.outputs, 0);
                        !output.has_value())
                    {
                        return std::unexpected(output.error());
                    }
                }
            }
            if (const auto* result = entry->find("result"); result != nullptr)
            {
                if (auto output = appendStageVaryings(*result, "varyingOutput", 0, stageInterface.outputs, 0);
                    !output.has_value())
                {
                    return std::unexpected(output.error());
                }
            }
            if (auto sorted = sortStageInterfaceEntries(stageInterface.inputs); !sorted.has_value())
            {
                return std::unexpected(sorted.error());
            }
            if (auto sorted = sortStageInterfaceEntries(stageInterface.outputs); !sorted.has_value())
            {
                return std::unexpected(sorted.error());
            }
            return stageInterface;
        }

        inline void appendStageInterfaceJson(std::string& text, std::span<const MetalStageInterfaceEntry> entries)
        {
            text += '[';
            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                if (index != 0)
                {
                    text += ", ";
                }
                const auto& entry = entries[index];
                text += "{\"location\": ";
                text += std::to_string(entry.location);
                text += ", \"component\": ";
                text += std::to_string(entry.component);
                text += ", \"componentType\": ";
                text += escapeJsonString(entry.componentType);
                text += ", \"componentCount\": ";
                text += std::to_string(entry.componentCount);
                text += '}';
            }
            text += ']';
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<MetalStageInterface, std::string> collectMetalStageInterface(
        std::string_view metalReflectionJson, std::string_view entryPoint = {})
    {
        auto parsed = detail::JsonParser(metalReflectionJson).parseDocument();
        if (!parsed.has_value())
        {
            return std::unexpected("Metal reflection JSON parse failed: " + parsed.error());
        }
        return detail::collectMetalStageInterface(*parsed, entryPoint);
    }

    namespace detail
    {
        inline void appendMetalArgumentBufferJson(std::string& text,
                                                  const MetalArgumentBufferArray& array,
                                                  bool separator)
        {
            if (separator)
            {
                text += ", ";
            }
            text += "{\"name\": ";
            text += escapeJsonString(array.name);
            text += ", \"type\": \"";
            text += array.resourceType;
            text += "\", \"set\": " + std::to_string(array.set) + ", \"binding\": " + std::to_string(array.binding) +
                    ", \"index\": " + std::to_string(array.metalIndex) +
                    ", \"capacity\": " + std::to_string(array.capacity) + "}";
        }
    } // namespace detail

    [[nodiscard]] inline std::expected<std::string, std::string> enrichMetalReflectionJson(
        std::string_view rawMetalReflectionJson,
        std::span<const MetalLogicalParameterBinding> bindings,
        std::string_view entryPoint = {},
        std::span<const MetalArgumentBufferArray> argumentBuffers = {})
    {
        auto parsed = detail::JsonParser(rawMetalReflectionJson).parseDocument();
        if (!parsed.has_value())
        {
            return std::unexpected("Cooked Metal reflection JSON parse failed: " + parsed.error());
        }
        const auto* rootObject = parsed->asObject();
        if (rootObject == nullptr)
        {
            return std::unexpected("Cooked Metal reflection JSON root must be an object.");
        }
        if (parsed->find("rachisBindings") != nullptr || parsed->find("rachisStageInterface") != nullptr)
        {
            return std::unexpected("Cooked Metal reflection JSON already contains RachisRHI annotations.");
        }
        const auto stageInterface = detail::collectMetalStageInterface(*parsed, entryPoint);
        if (!stageInterface.has_value())
        {
            return std::unexpected(stageInterface.error());
        }

        std::size_t closeBrace = rawMetalReflectionJson.size();
        while (closeBrace > 0)
        {
            const char ch = rawMetalReflectionJson[closeBrace - 1];
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
            {
                --closeBrace;
            }
            else
            {
                break;
            }
        }
        if (closeBrace == 0 || rawMetalReflectionJson[closeBrace - 1] != '}')
        {
            return std::unexpected("Cooked Metal reflection JSON is missing closing object brace.");
        }
        const std::size_t insertPos = closeBrace - 1;

        std::string suffix;
        if (!rootObject->empty())
        {
            suffix += ",\n    ";
        }
        suffix += "\"rachisBindings\": [";
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            if (index != 0)
            {
                suffix += ", ";
            }
            const auto& entry = bindings[index];
            suffix += "{\"name\": ";
            suffix += detail::escapeJsonString(entry.name);
            if (entry.pushConstant)
            {
                suffix += ", \"pushConstant\": true}";
            }
            else if (entry.drawIndex)
            {
                suffix += ", \"drawIndex\": true}";
            }
            else
            {
                suffix += ", \"set\": ";
                suffix += std::to_string(entry.set);
                suffix += ", \"binding\": ";
                suffix += std::to_string(entry.binding);
                suffix += '}';
            }
        }
        suffix += ']';
        if (!argumentBuffers.empty())
        {
            suffix += ",\n    \"rachisArgumentBuffers\": [";
            for (std::size_t index = 0; index < argumentBuffers.size(); ++index)
            {
                detail::appendMetalArgumentBufferJson(suffix, argumentBuffers[index], index != 0);
            }
            suffix += ']';
        }
        if (stageInterface->reflected)
        {
            suffix += ",\n    \"rachisStageInterface\": {\"inputs\": ";
            detail::appendStageInterfaceJson(suffix, stageInterface->inputs);
            suffix += ", \"outputs\": ";
            detail::appendStageInterfaceJson(suffix, stageInterface->outputs);
            suffix += '}';
        }
        suffix += '\n';

        std::string enriched;
        enriched.reserve(rawMetalReflectionJson.size() + suffix.size() + 2);
        enriched.append(rawMetalReflectionJson.substr(0, insertPos));
        enriched.append(suffix);
        enriched.push_back('}');
        enriched.append(rawMetalReflectionJson.substr(closeBrace));
        return enriched;
    }

    [[nodiscard]] inline std::expected<std::string, std::string> generateEmbeddedBinaryHeader(
        std::span<const std::byte> data, std::string_view symbol)
    {
        if (!detail::isValidCppIdentifier(symbol))
        {
            return std::unexpected("The symbol must be a C++ identifier.");
        }

        constexpr std::array<char, 16> hexDigits{
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

        std::string text;
        text.reserve(160 + (symbol.size() * 3) + (data.size() * 6));
        text += "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n";
        text += "inline constexpr std::uint8_t ";
        text += symbol;
        text += "[] = {\n";

        if (data.empty())
        {
            text += "    0,\n";
        }
        else
        {
            for (std::size_t offset = 0; offset < data.size(); offset += 16)
            {
                text += "    ";
                const std::size_t lineEnd = std::min(offset + std::size_t{16}, data.size());
                for (std::size_t index = offset; index < lineEnd; ++index)
                {
                    if (index != offset)
                    {
                        text += ", ";
                    }
                    const auto value = std::to_integer<unsigned int>(data[index]);
                    text.push_back('0');
                    text.push_back('x');
                    text.push_back(hexDigits[(value >> 4U) & 0x0FU]);
                    text.push_back(hexDigits[value & 0x0FU]);
                }
                text += ",\n";
            }
        }

        text += "};\n";
        text += "inline constexpr std::size_t ";
        text += symbol;
        text += "Size = ";
        if (data.empty())
        {
            text += "0;\n";
        }
        else
        {
            text += "sizeof(";
            text += symbol;
            text += ");\n";
        }
        return text;
    }

    [[nodiscard]] inline std::expected<std::vector<std::byte>, std::string> packMetalShaderManifest(
        std::span<const std::byte> reflection, std::span<const std::byte> metallib)
    {
        if (metallib.size() < 4 || metallib[0] != std::byte{'M'} || metallib[1] != std::byte{'T'} ||
            metallib[2] != std::byte{'L'} || metallib[3] != std::byte{'B'})
        {
            return std::unexpected("metallib does not start with MTLB.");
        }

        constexpr std::size_t headerBytes = 16;
        constexpr auto maxSectionBytes = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
        if (reflection.size() > maxSectionBytes || metallib.size() > maxSectionBytes)
        {
            return std::unexpected("PNMS payload section exceeds 32-bit length limit.");
        }
        if (reflection.size() > std::numeric_limits<std::size_t>::max() - headerBytes ||
            metallib.size() > std::numeric_limits<std::size_t>::max() - headerBytes - reflection.size())
        {
            return std::unexpected("PNMS payload size overflows.");
        }

        std::vector<std::byte> payload(headerBytes + reflection.size() + metallib.size());
        payload[0] = std::byte{'P'};
        payload[1] = std::byte{'N'};
        payload[2] = std::byte{'M'};
        payload[3] = std::byte{'S'};
        detail::writeU32LittleEndian(std::span<std::byte, 4>{payload.data() + 4, 4}, 1U);
        detail::writeU32LittleEndian(std::span<std::byte, 4>{payload.data() + 8, 4},
                                     static_cast<std::uint32_t>(reflection.size()));
        detail::writeU32LittleEndian(std::span<std::byte, 4>{payload.data() + 12, 4},
                                     static_cast<std::uint32_t>(metallib.size()));

        if (!reflection.empty())
        {
            std::memcpy(payload.data() + headerBytes, reflection.data(), reflection.size());
        }
        std::memcpy(payload.data() + headerBytes + reflection.size(), metallib.data(), metallib.size());
        return payload;
    }

    namespace detail
    {

        struct MetalSourceSpan final
        {
            std::size_t begin = 0;
            std::size_t end = 0;
        };

        struct MetalAttributedParameter final
        {
            MetalSourceSpan declaration{};
            MetalSourceSpan type{};
            MetalSourceSpan name{};
        };

        [[nodiscard]] constexpr bool isMetalSourceSpace(char ch) noexcept
        {
            return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
        }

        [[nodiscard]] constexpr MetalSourceSpan trimMetalSourceSpan(std::string_view source,
                                                                    MetalSourceSpan span) noexcept
        {
            while (span.begin < span.end && isMetalSourceSpace(source[span.begin]))
            {
                ++span.begin;
            }
            while (span.end > span.begin && isMetalSourceSpace(source[span.end - 1]))
            {
                --span.end;
            }
            return span;
        }

        [[nodiscard]] constexpr std::string_view metalSourceText(std::string_view source, MetalSourceSpan span) noexcept
        {
            return source.substr(span.begin, span.end - span.begin);
        }

        [[nodiscard]] constexpr std::optional<std::size_t> findClosingMetalParenthesis(std::string_view source,
                                                                                       std::size_t open) noexcept
        {
            std::size_t depth = 0;
            for (std::size_t index = open; index < source.size(); ++index)
            {
                if (source[index] == '(')
                {
                    ++depth;
                }
                else if (source[index] == ')' && --depth == 0)
                {
                    return index;
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::vector<MetalSourceSpan> splitMetalParameters(std::string_view source,
                                                                               std::size_t open,
                                                                               std::size_t close)
        {
            std::vector<MetalSourceSpan> parameters;
            std::size_t depth = 0;
            std::size_t start = open + 1;
            for (std::size_t index = open + 1; index < close; ++index)
            {
                const char ch = source[index];
                if (ch == '(' || ch == '[' || ch == '<')
                {
                    ++depth;
                }
                else if ((ch == ')' || ch == ']' || ch == '>') && depth > 0)
                {
                    --depth;
                }
                else if (ch == ',' && depth == 0)
                {
                    parameters.push_back(trimMetalSourceSpan(source, {start, index}));
                    start = index + 1;
                }
            }
            const MetalSourceSpan last = trimMetalSourceSpan(source, {start, close});
            if (last.begin != last.end || !parameters.empty())
            {
                parameters.push_back(last);
            }
            return parameters;
        }

        [[nodiscard]] inline std::expected<std::optional<MetalAttributedParameter>, std::string>
        findMetalAttributedParameter(std::string_view source,
                                     std::span<const MetalSourceSpan> parameters,
                                     std::string_view attribute)
        {
            for (const MetalSourceSpan parameter : parameters)
            {
                const std::size_t attributeOffset = metalSourceText(source, parameter).find(attribute);
                if (attributeOffset == std::string_view::npos)
                {
                    continue;
                }
                const MetalSourceSpan declaration =
                    trimMetalSourceSpan(source, {parameter.begin, parameter.begin + attributeOffset});
                std::size_t nameBegin = declaration.end;
                while (nameBegin > declaration.begin && isAsciiIdentifierContinue(source[nameBegin - 1]))
                {
                    --nameBegin;
                }
                const MetalSourceSpan type = trimMetalSourceSpan(source, {declaration.begin, nameBegin});
                if (nameBegin == declaration.end || !isAsciiIdentifierStart(source[nameBegin]) ||
                    type.begin == type.end)
                {
                    return std::unexpected("Cannot read the Metal entry-point parameter '" +
                                           std::string(metalSourceText(source, parameter)) + "'.");
                }
                return MetalAttributedParameter{
                    .declaration = parameter,
                    .type = type,
                    .name = {nameBegin, declaration.end},
                };
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::string unusedMetalIdentifier(std::string_view source, std::string_view stem)
        {
            std::string candidate(stem);
            for (std::size_t suffix = 0; source.find(candidate) != std::string_view::npos; ++suffix)
            {
                candidate = std::string(stem) + "_" + std::to_string(suffix);
            }
            return candidate;
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<std::optional<std::string>, std::string> rebaseMetalVertexIds(
        std::string_view source)
    {
        constexpr std::string_view vertexFunctionAttribute = "[[vertex]]";
        std::string rebased(source);
        bool changed = false;
        std::size_t searchFrom = 0;
        for (std::size_t entry = rebased.find(vertexFunctionAttribute); entry != std::string::npos;
             entry = rebased.find(vertexFunctionAttribute, searchFrom))
        {
            const std::size_t open = rebased.find('(', entry);
            if (open == std::string::npos)
            {
                return std::unexpected("A Metal [[vertex]] function has no parameter list.");
            }
            const auto close = detail::findClosingMetalParenthesis(rebased, open);
            if (!close.has_value())
            {
                return std::unexpected("A Metal [[vertex]] function's parameter list is not closed.");
            }
            const std::size_t bodyOpen = rebased.find_first_not_of(" \t\r\n", *close + 1);
            if (bodyOpen == std::string::npos || rebased[bodyOpen] != '{')
            {
                return std::unexpected("A Metal [[vertex]] function's body does not follow its parameter list.");
            }
            searchFrom = bodyOpen + 1;

            const auto parameters = detail::splitMetalParameters(rebased, open, *close);
            const auto vertexId = detail::findMetalAttributedParameter(rebased, parameters, "[[vertex_id]]");
            if (!vertexId.has_value())
            {
                return std::unexpected(vertexId.error());
            }
            if (!vertexId->has_value())
            {
                continue;
            }
            const auto baseVertex = detail::findMetalAttributedParameter(rebased, parameters, "[[base_vertex]]");
            if (!baseVertex.has_value())
            {
                return std::unexpected(baseVertex.error());
            }

            const detail::MetalAttributedParameter& vertexIdParameter = **vertexId;
            const std::string rawName = detail::unusedMetalIdentifier(rebased, "rachis_vertex_id");
            std::string baseName;
            std::string addedParameter;
            if (baseVertex->has_value())
            {
                baseName = detail::metalSourceText(rebased, (*baseVertex)->name);
            }
            else
            {
                baseName = detail::unusedMetalIdentifier(rebased, "rachis_base_vertex");
                addedParameter = ", uint " + baseName + " [[base_vertex]]";
            }
            const std::string zeroBasedDeclaration =
                "\n    " + std::string(detail::metalSourceText(rebased, vertexIdParameter.type)) + " " +
                std::string(detail::metalSourceText(rebased, vertexIdParameter.name)) + " = " + rawName + " - " +
                baseName + ";";

            rebased.insert(bodyOpen + 1, zeroBasedDeclaration);
            rebased.insert(vertexIdParameter.declaration.end, addedParameter);
            const std::size_t nameLength = vertexIdParameter.name.end - vertexIdParameter.name.begin;
            rebased.replace(vertexIdParameter.name.begin, nameLength, rawName);
            searchFrom += zeroBasedDeclaration.size() + addedParameter.size() + rawName.size() - nameLength;
            changed = true;
        }
        if (!changed)
        {
            return std::optional<std::string>{};
        }
        return std::optional<std::string>{std::move(rebased)};
    }

    namespace detail
    {

        inline constexpr std::array<std::string_view, 5> metalEntryAttributes{
            "[[kernel]]",
            "[[vertex]]",
            "[[fragment]]",
            "[[object]]",
            "[[mesh]]",
        };

        struct MetalSourceRange final
        {
            std::size_t begin = 0;
            std::size_t end = 0;
        };

        [[nodiscard]] inline std::optional<std::size_t> matchingParenthesis(std::string_view source,
                                                                            std::size_t open) noexcept
        {
            std::size_t depth = 0;
            for (std::size_t index = open; index < source.size(); ++index)
            {
                if (source[index] == '(')
                {
                    ++depth;
                }
                else if (source[index] == ')')
                {
                    if (--depth == 0)
                    {
                        return index;
                    }
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::vector<MetalSourceRange> metalEntryParameterLists(std::string_view source)
        {
            std::vector<MetalSourceRange> ranges;
            for (const auto attribute : metalEntryAttributes)
            {
                for (std::size_t found = source.find(attribute); found != std::string_view::npos;
                     found = source.find(attribute, found + attribute.size()))
                {
                    const auto open = source.find('(', found);
                    if (open == std::string_view::npos)
                    {
                        continue;
                    }
                    if (const auto close = matchingParenthesis(source, open); close.has_value())
                    {
                        ranges.push_back(MetalSourceRange{open, *close});
                    }
                }
            }
            return ranges;
        }

        [[nodiscard]] inline bool isMetalIdentifierChar(char ch) noexcept
        {
            return isAsciiIdentifierContinue(ch);
        }

        struct MetalUnboundedDeclaration final
        {
            std::size_t typeBegin = 0;
            std::size_t nameBegin = 0;
            std::size_t end = 0;
            std::string type;
            std::string name;
        };

        [[nodiscard]] inline std::optional<MetalUnboundedDeclaration> metalUnboundedDeclarationAt(
            std::string_view source, std::size_t brackets)
        {
            std::size_t nameEnd = brackets;
            std::size_t nameBegin = nameEnd;
            while (nameBegin > 0 && isMetalIdentifierChar(source[nameBegin - 1]))
            {
                --nameBegin;
            }
            if (nameBegin == nameEnd || !isAsciiIdentifierStart(source[nameBegin]))
            {
                return std::nullopt;
            }
            std::size_t typeEnd = nameBegin;
            while (typeEnd > 0 && source[typeEnd - 1] == ' ')
            {
                --typeEnd;
            }
            if (typeEnd == nameBegin)
            {
                return std::nullopt;
            }
            std::size_t typeBegin = typeEnd;
            int depth = 0;
            while (typeBegin > 0)
            {
                const char ch = source[typeBegin - 1];
                if (ch == '>')
                {
                    ++depth;
                }
                else if (ch == '<')
                {
                    --depth;
                }
                else if (depth == 0 && (ch == ',' || ch == '(' || ch == '{' || ch == ';' || ch == '\n'))
                {
                    break;
                }
                --typeBegin;
            }
            while (typeBegin < typeEnd && (source[typeBegin] == ' ' || source[typeBegin] == '\t'))
            {
                ++typeBegin;
            }
            if (typeBegin == typeEnd)
            {
                return std::nullopt;
            }
            return MetalUnboundedDeclaration{
                .typeBegin = typeBegin,
                .nameBegin = nameBegin,
                .end = brackets + 2,
                .type = std::string(source.substr(typeBegin, typeEnd - typeBegin)),
                .name = std::string(source.substr(nameBegin, nameEnd - nameBegin)),
            };
        }

        [[nodiscard]] inline std::vector<MetalUnboundedDeclaration> metalUnboundedDeclarations(std::string_view source)
        {
            std::vector<MetalUnboundedDeclaration> declarations;
            for (std::size_t found = source.find("[]"); found != std::string_view::npos;
                 found = source.find("[]", found + 2))
            {
                if (auto declaration = metalUnboundedDeclarationAt(source, found); declaration.has_value())
                {
                    declarations.push_back(std::move(*declaration));
                }
            }
            return declarations;
        }

        [[nodiscard]] inline std::string metalArrayBaseName(std::string_view parameterName)
        {
            const auto underscore = parameterName.rfind('_');
            if (underscore == std::string_view::npos || underscore + 1 == parameterName.size() ||
                !std::ranges::all_of(parameterName.substr(underscore + 1),
                                     [](char ch) { return ch >= '0' && ch <= '9'; }))
            {
                return std::string(parameterName);
            }
            return std::string(parameterName.substr(0, underscore));
        }

        [[nodiscard]] inline std::optional<std::uint32_t> highestMetalBufferIndex(std::string_view source)
        {
            constexpr std::string_view attribute = "[[buffer(";
            std::optional<std::uint32_t> highest;
            for (std::size_t found = source.find(attribute); found != std::string_view::npos;
                 found = source.find(attribute, found + attribute.size()))
            {
                std::uint32_t value = 0;
                std::size_t digit = found + attribute.size();
                bool any = false;
                while (digit < source.size() && source[digit] >= '0' && source[digit] <= '9')
                {
                    value = value * 10U + static_cast<std::uint32_t>(source[digit] - '0');
                    any = true;
                    ++digit;
                }
                if (any)
                {
                    highest = highest.has_value() ? std::max(*highest, value) : value;
                }
            }
            return highest;
        }

        [[nodiscard]] inline const JsonValue* findRootParameter(const JsonValue& root, std::string_view name)
        {
            const auto* parameters = root.find("parameters");
            if (parameters == nullptr || parameters->asArray() == nullptr)
            {
                return nullptr;
            }
            for (const auto& parameter : *parameters->asArray())
            {
                if (const auto* parameterName = findString(parameter, "name");
                    parameterName != nullptr && *parameterName == name)
                {
                    return &parameter;
                }
            }
            return nullptr;
        }

        [[nodiscard]] inline std::optional<std::string_view> metalArgumentBufferResourceType(const JsonValue& parameter)
        {
            const auto* type = parameter.find("type");
            if (type == nullptr || findString(*type, "kind") == nullptr || *findString(*type, "kind") != "array")
            {
                return std::nullopt;
            }
            const auto* element = type->find("elementType");
            const auto* kind = element != nullptr ? findString(*element, "kind") : nullptr;
            if (kind == nullptr)
            {
                return std::nullopt;
            }
            if (*kind == "constantBuffer")
            {
                return std::string_view{"uniformBuffer"};
            }
            if (*kind == "samplerState")
            {
                return std::string_view{"sampler"};
            }
            if (*kind != "resource")
            {
                return std::nullopt;
            }
            const auto* shape = findString(*element, "baseShape");
            const auto* access = findString(*element, "access");
            const bool writable = access != nullptr && *access == "readWrite";
            if (shape != nullptr && shape->starts_with("texture"))
            {
                return writable ? std::string_view{"storageTexture"} : std::string_view{"sampledTexture"};
            }
            if (shape != nullptr && (*shape == "structuredBuffer" || *shape == "byteAddressBuffer"))
            {
                return writable ? std::string_view{"storageBuffer"} : std::string_view{"readOnlyStorageBuffer"};
            }
            return std::nullopt;
        }

        struct MetalLogicalSlot final
        {
            std::uint32_t set = 0;
            std::uint32_t binding = 0;
        };

        [[nodiscard]] inline std::optional<MetalLogicalSlot> spirvDescriptorSlot(const JsonValue& parameter)
        {
            const auto* binding = parameter.find("binding");
            if (binding == nullptr || findString(*binding, "kind") == nullptr ||
                *findString(*binding, "kind") != "descriptorTableSlot")
            {
                return std::nullopt;
            }
            const auto index = findUint32(*binding, "index");
            const auto space =
                binding->find("space") != nullptr ? findUint32(*binding, "space") : std::optional<std::uint32_t>{0};
            if (!index.has_value() || !space.has_value())
            {
                return std::nullopt;
            }
            return MetalLogicalSlot{.set = *space, .binding = *index};
        }

    } // namespace detail

    [[nodiscard]] inline std::vector<MetalUnboundedArrayParameter> collectMetalUnboundedArrayParameters(
        std::string_view source)
    {
        std::vector<MetalUnboundedArrayParameter> parameters;
        const auto entries = detail::metalEntryParameterLists(source);
        for (const auto& declaration : detail::metalUnboundedDeclarations(source))
        {
            const bool inEntry = std::ranges::any_of(entries, [&](const detail::MetalSourceRange& range) {
                return declaration.nameBegin > range.begin && declaration.end <= range.end;
            });
            if (!inEntry || std::ranges::any_of(parameters, [&](const MetalUnboundedArrayParameter& existing) {
                    return existing.parameterName == declaration.name;
                }))
            {
                continue;
            }
            parameters.push_back(MetalUnboundedArrayParameter{
                .parameterName = declaration.name,
                .baseName = detail::metalArrayBaseName(declaration.name),
                .elementType = declaration.type,
            });
        }
        return parameters;
    }

    [[nodiscard]] inline std::expected<std::vector<MetalArgumentBufferArray>, std::string>
    planMetalArgumentBufferArrays(std::string_view metalSource,
                                  std::string_view spirvReflectionJson,
                                  std::string_view metalReflectionJson)
    {
        const auto parameters = collectMetalUnboundedArrayParameters(metalSource);
        std::vector<MetalArgumentBufferArray> arrays;
        if (parameters.empty())
        {
            return arrays;
        }
        auto spirvRoot = detail::JsonParser(spirvReflectionJson).parseDocument();
        auto metalRoot = detail::JsonParser(metalReflectionJson).parseDocument();
        if (!spirvRoot.has_value() || !metalRoot.has_value())
        {
            return std::unexpected("Metal argument-buffer planning could not parse the reflection JSON.");
        }
        const auto highest = detail::highestMetalBufferIndex(metalSource);
        std::uint32_t nextIndex = highest.has_value() ? *highest + 1U : 0U;
        for (const auto& parameter : parameters)
        {
            const auto* metalParameter = detail::findRootParameter(*metalRoot, parameter.baseName);
            const auto* spirvParameter = detail::findRootParameter(*spirvRoot, parameter.baseName);
            if (metalParameter == nullptr || spirvParameter == nullptr)
            {
                return std::unexpected("The unbounded Metal resource array '" + parameter.baseName +
                                       "' is missing from the reflection.");
            }
            const auto resourceType = detail::metalArgumentBufferResourceType(*metalParameter);
            const auto slot = detail::spirvDescriptorSlot(*spirvParameter);
            if (!resourceType.has_value() || !slot.has_value())
            {
                return std::unexpected("The unbounded Metal resource array '" + parameter.baseName +
                                       "' has an unsupported element type or binding.");
            }
            if (nextIndex >= metalArgumentTableBufferSlots)
            {
                return std::unexpected("Unbounded resource arrays exceed Metal's 31 buffer argument slots.");
            }
            arrays.push_back(MetalArgumentBufferArray{
                .name = parameter.baseName,
                .parameterName = parameter.parameterName,
                .elementType = parameter.elementType,
                .resourceType = *resourceType,
                .set = slot->set,
                .binding = slot->binding,
                .metalIndex = nextIndex++,
                .capacity =
                    *resourceType == "sampler" ? metalArgumentBufferSamplerLimit : metalArgumentBufferResourceLimit,
            });
        }
        return arrays;
    }

    [[nodiscard]] inline std::expected<std::string, std::string> rewriteMetalArgumentBufferArrays(
        std::string_view source, std::span<const MetalArgumentBufferArray> arrays)
    {
        if (arrays.empty())
        {
            return std::string(source);
        }
        constexpr std::string_view usingNamespace = "using namespace metal;\n";
        const auto insertAt = source.find(usingNamespace);
        if (insertAt == std::string_view::npos)
        {
            return std::unexpected("Generated Metal source has no 'using namespace metal;' line.");
        }
        const auto capacityOf = [&](std::string_view type) -> std::optional<std::uint32_t> {
            for (const auto& array : arrays)
            {
                if (array.elementType == type)
                {
                    return array.capacity;
                }
            }
            return std::nullopt;
        };
        const auto entries = detail::metalEntryParameterLists(source);
        std::string rewritten;
        rewritten.reserve(source.size() + 1024);
        std::size_t copied = 0;
        for (const auto& declaration : detail::metalUnboundedDeclarations(source))
        {
            const auto capacity = capacityOf(declaration.type);
            if (!capacity.has_value())
            {
                return std::unexpected("Generated Metal source declares an unbounded array of '" + declaration.type +
                                       "' that the argument-buffer plan does not cover.");
            }
            const auto parameter =
                std::ranges::find(arrays, declaration.name, &MetalArgumentBufferArray::parameterName);
            const bool inEntry = std::ranges::any_of(entries, [&](const detail::MetalSourceRange& range) {
                return declaration.nameBegin > range.begin && declaration.end <= range.end;
            });
            rewritten.append(source.substr(copied, declaration.typeBegin - copied));
            const auto templateArguments = declaration.type + ", " + std::to_string(*capacity);
            if (inEntry && parameter != arrays.end())
            {
                rewritten += "constant RachisArgumentStorage<" + templateArguments + ">& " + declaration.name +
                             " [[buffer(" + std::to_string(parameter->metalIndex) + ")]]";
            }
            else
            {
                rewritten += "RachisArgumentArray<" + templateArguments + "> " + declaration.name;
            }
            copied = declaration.end;
        }
        rewritten.append(source.substr(copied));

        std::string assigned;
        assigned.reserve(rewritten.size());
        std::size_t lineBegin = 0;
        while (lineBegin < rewritten.size())
        {
            auto lineEnd = rewritten.find('\n', lineBegin);
            if (lineEnd == std::string::npos)
            {
                lineEnd = rewritten.size();
            }
            std::string_view line(rewritten.data() + lineBegin, lineEnd - lineBegin);
            bool replaced = false;
            for (const auto& array : arrays)
            {
                const auto suffix = " = " + array.parameterName + ";";
                if (line.ends_with(suffix) && line.size() > suffix.size())
                {
                    const auto target = line.substr(0, line.size() - suffix.size());
                    assigned.append(target);
                    assigned += ".items = &" + array.parameterName + ";";
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
            {
                assigned.append(line);
            }
            if (lineEnd < rewritten.size())
            {
                assigned.push_back('\n');
            }
            lineBegin = lineEnd + 1;
        }

        constexpr std::string_view helpers = "template<typename T, uint N>\n"
                                             "struct RachisArgumentStorage\n"
                                             "{\n"
                                             "    metal::array<T, N> items;\n"
                                             "    T operator[](uint index) const constant\n"
                                             "    {\n"
                                             "        return items[index];\n"
                                             "    }\n"
                                             "};\n"
                                             "template<typename T, uint N>\n"
                                             "struct RachisArgumentArray\n"
                                             "{\n"
                                             "    constant RachisArgumentStorage<T, N>* items;\n"
                                             "    T operator[](uint index) const thread\n"
                                             "    {\n"
                                             "        return items->items[index];\n"
                                             "    }\n"
                                             "};\n";
        const auto helperAt = assigned.find(usingNamespace);
        assigned.insert(helperAt + usingNamespace.size(), helpers);
        return assigned;
    }

} // namespace rachis::shader_tool
