#include "Common/HandleTable.h"
#include "Common/MonotonicIdentitySource.h"
#include "Rachis/Error.h"
#include "Rachis/Handle.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

namespace
{

    struct TestTag;
    using TestHandle = rachis::Handle<TestTag>;

    struct OtherTag;
    using OtherHandle = rachis::Handle<OtherTag>;

} // namespace

TEST_CASE("slot-generation handles reject stale and foreign tables", "[size:small][rhi][handle]")
{
    REQUIRE_FALSE(TestHandle{}.isValid());

    rachis::detail::HandleTable<std::string, TestHandle> left(11);
    rachis::detail::HandleTable<std::string, TestHandle> right(12);
    rachis::detail::HandleTable<std::string, OtherHandle> otherKind(13);

    const auto firstResult = left.emplace("first");
    REQUIRE(firstResult.has_value());
    const auto first = *firstResult;
    REQUIRE(first.isValid());
    REQUIRE(left.find(first) != nullptr);
    CHECK(*left.find(first) == "first");

    const auto foreignResult = right.emplace("foreign");
    REQUIRE(foreignResult.has_value());
    const auto foreign = *foreignResult;
    CHECK(foreign.identity != first.identity);
    CHECK(right.find(first) == nullptr);
    CHECK(left.find(foreign) == nullptr);

    const auto otherResult = otherKind.emplace("other-kind");
    REQUIRE(otherResult.has_value());
    CHECK(otherResult->identity != first.identity);

    REQUIRE(left.erase(first));
    CHECK(left.find(first) == nullptr);
    CHECK_FALSE(left.erase(first));

    const auto reusedResult = left.emplace("reused");
    REQUIRE(reusedResult.has_value());
    const auto reused = *reusedResult;
    CHECK(reused.identity != first.identity);
    CHECK(left.find(first) == nullptr);

    const auto secondResult = left.emplace("second");
    REQUIRE(secondResult.has_value());
    const auto second = *secondResult;
    CHECK(left.size() == 2);
    left.clear();
    CHECK(left.size() == 0);
    CHECK(left.find(reused) == nullptr);
    CHECK(left.find(second) == nullptr);

    const auto afterClearResult = left.emplace("after-clear");
    REQUIRE(afterClearResult.has_value());
    CHECK(left.size() == 1);
    CHECK(*afterClearResult != reused);
    CHECK(*afterClearResult != second);
}

TEST_CASE("default owner tokens are shared by raw handle type", "[size:small][rhi][handle]")
{
    rachis::detail::HandleTable<std::string, TestHandle> text;
    rachis::detail::HandleTable<std::uint32_t, TestHandle> number;

    const auto textHandle = text.emplace("text").value();
    const auto numberHandle = number.emplace(42U).value();

    CHECK(textHandle != numberHandle);
    CHECK(text.find(numberHandle) == nullptr);
    CHECK(number.find(textHandle) == nullptr);
}

TEST_CASE("owner tokens outside the 24-bit owner field never alias another table", "[size:small][rhi][handle]")
{
    rachis::detail::HandleTable<std::string, TestHandle> lowest(1U);
    const auto lowestHandle = lowest.emplace("lowest");
    REQUIRE(lowestHandle.has_value());

    rachis::detail::HandleTable<std::string, TestHandle> zero(0U);
    rachis::detail::HandleTable<std::string, TestHandle> wrapped(0x0100'0000U);
    rachis::detail::HandleTable<std::string, TestHandle> wrappedNext(0x0100'0001U);
    for (auto* table : {&zero, &wrapped, &wrappedNext})
    {
        const auto refused = table->emplace("wrapped");
        REQUIRE_FALSE(refused.has_value());
        CHECK(refused.error().code == rachis::ErrorCode::resourceExhausted);
        CHECK(table->size() == 0);
        CHECK(table->find(*lowestHandle) == nullptr);
        CHECK_FALSE(table->erase(*lowestHandle));
    }
    REQUIRE(lowest.find(*lowestHandle) != nullptr);
    CHECK(*lowest.find(*lowestHandle) == "lowest");
}

TEST_CASE("handle tables never reuse an owner token after the owner space runs out", "[size:small][rhi][handle]")
{
    rachis::detail::MonotonicIdentitySource ownerTokens(0xFF'FFFE);
    rachis::detail::HandleTable<std::string, TestHandle> penultimate(ownerTokens);
    rachis::detail::HandleTable<std::string, TestHandle> last(ownerTokens);
    rachis::detail::HandleTable<std::string, TestHandle> exhausted(ownerTokens);
    rachis::detail::HandleTable<std::string, TestHandle> stillExhausted(ownerTokens);

    const auto penultimateHandle = penultimate.emplace("penultimate");
    const auto lastHandle = last.emplace("last");
    REQUIRE(penultimateHandle.has_value());
    REQUIRE(lastHandle.has_value());
    CHECK(*penultimateHandle != *lastHandle);
    CHECK(penultimate.find(*lastHandle) == nullptr);
    CHECK(last.find(*penultimateHandle) == nullptr);

    for (auto* table : {&exhausted, &stillExhausted})
    {
        const auto refused = table->emplace("refused");
        REQUIRE_FALSE(refused.has_value());
        CHECK(refused.error().code == rachis::ErrorCode::resourceExhausted);
        CHECK(table->size() == 0);
        CHECK(table->find(*penultimateHandle) == nullptr);
        CHECK(table->find(*lastHandle) == nullptr);
    }
    REQUIRE(last.find(*lastHandle) != nullptr);
    CHECK(*last.find(*lastHandle) == "last");
}

TEST_CASE("handle tables cannot be copied", "[size:small][rhi][handle]")
{
    using Table = rachis::detail::HandleTable<std::string, TestHandle>;
    STATIC_CHECK_FALSE(std::is_copy_constructible_v<Table>);
    STATIC_CHECK_FALSE(std::is_copy_assignable_v<Table>);
    STATIC_CHECK(std::is_move_constructible_v<Table>);
    STATIC_CHECK(std::is_nothrow_move_assignable_v<Table>);
}

TEST_CASE("moving a handle table hands over its handles and leaves an empty table that issues none",
          "[size:small][rhi][handle]")
{
    rachis::detail::HandleTable<std::string, TestHandle> source(41);
    const auto erased = source.emplace("erased");
    const auto kept = source.emplace("kept");
    REQUIRE(erased.has_value());
    REQUIRE(kept.has_value());
    REQUIRE(source.erase(*erased));
    const auto freeCapacity = source.freeListCapacity();

    rachis::detail::HandleTable<std::string, TestHandle> moved(std::move(source));
    CHECK(moved.size() == 1);
    CHECK(moved.freeListCapacity() == freeCapacity);
    REQUIRE(moved.find(*kept) != nullptr);
    CHECK(*moved.find(*kept) == "kept");
    CHECK(moved.find(*erased) == nullptr);

    CHECK(source.size() == 0);
    CHECK(source.find(*kept) == nullptr);
    CHECK_FALSE(source.erase(*kept));
    std::size_t visited = 0;
    source.forEach([&](TestHandle, std::string&) { ++visited; });
    CHECK(visited == 0);
    const auto intruder = source.emplace("intruder");
    REQUIRE_FALSE(intruder.has_value());
    CHECK(intruder.error().code == rachis::ErrorCode::resourceExhausted);

    const auto reused = moved.emplace("reused");
    REQUIRE(reused.has_value());
    CHECK(*reused != *erased);
    CHECK(*reused != *kept);
    CHECK(moved.size() == 2);
    REQUIRE(moved.find(*kept) != nullptr);
    CHECK(*moved.find(*kept) == "kept");
}

TEST_CASE("move-assigning a handle table retires the target's handles", "[size:small][rhi][handle]")
{
    rachis::detail::HandleTable<std::string, TestHandle> target(51);
    rachis::detail::HandleTable<std::string, TestHandle> source(52);
    const auto replaced = target.emplace("replaced");
    const auto kept = source.emplace("kept");
    REQUIRE(replaced.has_value());
    REQUIRE(kept.has_value());

    target = std::move(source);
    CHECK(target.size() == 1);
    CHECK(target.find(*replaced) == nullptr);
    REQUIRE(target.find(*kept) != nullptr);
    CHECK(*target.find(*kept) == "kept");

    CHECK(source.size() == 0);
    CHECK(source.find(*kept) == nullptr);
    const auto intruder = source.emplace("intruder");
    REQUIRE_FALSE(intruder.has_value());
    CHECK(intruder.error().code == rachis::ErrorCode::resourceExhausted);

    rachis::detail::HandleTable<std::string, TestHandle> fresh(53);
    std::swap(fresh, target);
    CHECK(fresh.size() == 1);
    REQUIRE(fresh.find(*kept) != nullptr);
    CHECK(*fresh.find(*kept) == "kept");
    CHECK(target.size() == 0);
    CHECK(target.find(*kept) == nullptr);
    const auto afterSwap = target.emplace("after-swap");
    REQUIRE(afterSwap.has_value());
    CHECK(fresh.find(*afterSwap) == nullptr);
    CHECK(target.find(*kept) == nullptr);
}

TEST_CASE("generation exhaustion retires only the exhausted slot", "[size:medium][rhi][handle]")
{
    rachis::detail::HandleTable<std::string, TestHandle> table(21);
    auto current = table.emplace("value").value();
    const auto first = current;

    for (std::uint32_t generation = 1; generation < std::numeric_limits<std::uint16_t>::max(); ++generation)
    {
        REQUIRE(table.erase(current));
        current = table.emplace("value").value();
    }

    REQUIRE(table.erase(current));
    const auto replacement = table.emplace("replacement").value();
    CHECK(replacement != current);
    CHECK(replacement != first);
    CHECK(table.find(current) == nullptr);
    CHECK(table.find(first) == nullptr);
    REQUIRE(table.find(replacement) != nullptr);
    CHECK(*table.find(replacement) == "replacement");
}

TEST_CASE("forEach visits only live table slots", "[size:small][rhi][handle]")
{
    rachis::detail::HandleTable<std::string, TestHandle> table(31);
    const auto first = table.emplace("first").value();
    const auto second = table.emplace("second").value();
    REQUIRE(table.erase(first));

    std::size_t visited = 0;
    table.forEach([&](TestHandle handle, std::string& value) {
        ++visited;
        CHECK(handle == second);
        CHECK(value == "second");
    });
    CHECK(visited == 1);
}
