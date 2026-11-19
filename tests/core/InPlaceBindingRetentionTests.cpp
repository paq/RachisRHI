#include "Common/InPlaceBindingRetention.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace
{

    using Retention = rachis::detail::InPlaceBindingRetention<std::shared_ptr<int>>;

    [[nodiscard]] std::vector<int> valuesSince(const Retention& retention, std::uint64_t sequence)
    {
        std::vector<int> values;
        retention.visitSince(sequence, [&](const std::shared_ptr<int>& element) {
            values.push_back(element != nullptr ? *element : -1);
        });
        return values;
    }

    [[nodiscard]] std::uint64_t recordWrites(Retention& retention, std::span<Retention::Write> writes)
    {
        REQUIRE(retention.reserve(writes).has_value());
        return retention.record(writes);
    }

} // namespace

TEST_CASE("update-after-bind retention keeps only the latest element of each index", "[size:small][rhi][binding]")
{
    Retention retention;
    auto first = std::make_shared<int>(1);
    const std::weak_ptr<int> firstObserver = first;

    std::array initial{
        Retention::Write{.index = 4, .element = std::move(first)},
        Retention::Write{.index = 9, .element = std::make_shared<int>(2)},
    };
    CHECK(recordWrites(retention, initial) == 1);
    CHECK(retention.contains(4));
    CHECK(retention.contains(9));
    CHECK_FALSE(retention.contains(5));
    CHECK(valuesSince(retention, 0) == std::vector{2, 1});

    std::array replacement{Retention::Write{.index = 4, .element = std::make_shared<int>(3)}};
    CHECK(recordWrites(retention, replacement) == 2);
    CHECK(firstObserver.expired());
    CHECK(valuesSince(retention, 0) == std::vector{3, 2});
    CHECK(valuesSince(retention, 1) == std::vector{3});
    CHECK(valuesSince(retention, 2).empty());

    std::array cleared{Retention::Write{.index = 9, .element = nullptr}};
    CHECK(recordWrites(retention, cleared) == 3);
    CHECK(valuesSince(retention, 2) == std::vector{-1});
    CHECK(valuesSince(retention, 0) == std::vector{-1, 3});
    CHECK(retention.sequence() == 3);
}

TEST_CASE("update-after-bind retention ignores empty updates and unrecorded reservations", "[size:small][rhi][binding]")
{
    Retention retention;
    CHECK(retention.record({}) == 0);
    CHECK(retention.sequence() == 0);

    std::array reserved{Retention::Write{.index = 7, .element = std::make_shared<int>(7)}};
    REQUIRE(retention.reserve(reserved).has_value());
    CHECK_FALSE(retention.contains(7));
    CHECK(valuesSince(retention, 0).empty());

    CHECK(retention.record(reserved) == 1);
    CHECK(retention.contains(7));
    CHECK(valuesSince(retention, 0) == std::vector{7});
}
