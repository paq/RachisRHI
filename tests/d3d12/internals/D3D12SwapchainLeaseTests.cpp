#include "D3D12/D3D12SwapchainLease.h"
#include "Internal/TimelinePointAccess.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>

#ifdef _WINDOWS_
#error Portable swapchain lease unit tests must not include Windows.h.
#endif

namespace
{

    [[nodiscard]] constexpr rachis::TimelinePoint makePoint(rachis::QueueType queueType,
                                                            std::uint64_t value,
                                                            std::uint64_t deviceToken = 1) noexcept
    {
        return rachis::detail::TimelinePointAccess::make(queueType, value, deviceToken);
    }

} // namespace

TEST_CASE("D3D12 swapchain lease supports concurrent multi-submit and requires the latest timeline point",
          "[size:small][rhi][d3d12][swapchain][lease]")
{
    auto lease = std::make_shared<rachis::d3d12::SwapchainImageLeaseState>();
    REQUIRE(lease->acquire().has_value());
    CHECK_FALSE(lease->acquire().has_value());

    auto first = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    auto second = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    const auto point1 = makePoint(rachis::QueueType::graphics, 10);
    const auto point2 = makePoint(rachis::QueueType::graphics, 11);

    auto cancelWhileRecorded = lease->cancel();
    REQUIRE_FALSE(cancelWhileRecorded.has_value());
    CHECK(cancelWhileRecorded.error().code == rachis::ErrorCode::invalidState);

    auto presentBeforeSubmit = lease->validatePresentationPoint(point1);
    REQUIRE_FALSE(presentBeforeSubmit.has_value());
    CHECK(presentBeforeSubmit.error().code == rachis::ErrorCode::invalidState);

    REQUIRE((*first)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*first)->markSubmitted().has_value());
    REQUIRE((*first)->publishSubmissionPoint(point1).has_value());

    auto presentWithOpenSecond = lease->validatePresentationPoint(point1);
    REQUIRE_FALSE(presentWithOpenSecond.has_value());
    CHECK(presentWithOpenSecond.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(lease->beginPresentation(point1).has_value());
    CHECK_FALSE(lease->failPresentation(point1).has_value());
    CHECK_FALSE(lease->cancel().has_value());
    CHECK_FALSE(lease->acquire().has_value());

    REQUIRE((*second)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*second)->markSubmitted().has_value());
    REQUIRE((*second)->publishSubmissionPoint(point2).has_value());

    auto stalePresent = lease->validatePresentationPoint(point1);
    REQUIRE_FALSE(stalePresent.has_value());
    CHECK(stalePresent.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(lease->beginPresentation(point1).has_value());

    REQUIRE(lease->validatePresentationPoint(point2).has_value());
    REQUIRE(lease->beginPresentation(point2).has_value());

    REQUIRE(lease->acquire().has_value());
    REQUIRE(lease->cancel().has_value());
}

TEST_CASE("D3D12 swapchain lease supports sequential record-submit-record-submit and single-batch multi-submit",
          "[size:small][rhi][d3d12][swapchain][lease]")
{
    auto lease = std::make_shared<rachis::d3d12::SwapchainImageLeaseState>();
    REQUIRE(lease->acquire().has_value());

    const auto pass1Point = makePoint(rachis::QueueType::graphics, 21);
    const auto pass2Point = makePoint(rachis::QueueType::graphics, 22);

    auto pass1 = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(pass1.has_value());
    REQUIRE((*pass1)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*pass1)->markSubmitted().has_value());
    REQUIRE((*pass1)->publishSubmissionPoint(pass1Point).has_value());
    REQUIRE(lease->validatePresentationPoint(pass1Point).has_value());

    auto pass2 = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(pass2.has_value());
    auto prematurePresent = lease->validatePresentationPoint(pass1Point);
    REQUIRE_FALSE(prematurePresent.has_value());
    CHECK(prematurePresent.error().code == rachis::ErrorCode::invalidState);

    REQUIRE((*pass2)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*pass2)->markSubmitted().has_value());
    REQUIRE((*pass2)->publishSubmissionPoint(pass2Point).has_value());

    auto stalePass1Present = lease->validatePresentationPoint(pass1Point);
    REQUIRE_FALSE(stalePass1Present.has_value());
    CHECK(stalePass1Present.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(lease->validatePresentationPoint(pass2Point).has_value());
    REQUIRE(lease->beginPresentation(pass2Point).has_value());

    REQUIRE(lease->acquire().has_value());
    auto batchFirst = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    auto batchSecond = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(batchFirst.has_value());
    REQUIRE(batchSecond.has_value());

    const auto batchPoint = makePoint(rachis::QueueType::graphics, 30);
    REQUIRE((*batchFirst)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*batchSecond)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*batchFirst)->markSubmitted().has_value());
    REQUIRE((*batchSecond)->markSubmitted().has_value());

    CHECK_FALSE(lease->validatePresentationPoint(batchPoint).has_value());
    REQUIRE((*batchFirst)->publishSubmissionPoint(batchPoint).has_value());

    CHECK_FALSE(lease->validatePresentationPoint(batchPoint).has_value());
    REQUIRE((*batchSecond)->publishSubmissionPoint(batchPoint).has_value());

    REQUIRE(lease->validatePresentationPoint(batchPoint).has_value());
    REQUIRE(lease->beginPresentation(batchPoint).has_value());
}

TEST_CASE("D3D12 swapchain lease handles abandoned second recordings and blocks unfinalized submissions",
          "[size:small][rhi][d3d12][swapchain][lease]")
{
    auto lease = std::make_shared<rachis::d3d12::SwapchainImageLeaseState>();
    REQUIRE(lease->acquire().has_value());

    const auto point1 = makePoint(rachis::QueueType::graphics, 40);
    auto submittedRef = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    auto abandonedRef = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(submittedRef.has_value());
    REQUIRE(abandonedRef.has_value());

    REQUIRE((*submittedRef)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*submittedRef)->markSubmitted().has_value());
    REQUIRE((*submittedRef)->publishSubmissionPoint(point1).has_value());

    CHECK_FALSE(lease->validatePresentationPoint(point1).has_value());

    abandonedRef->reset();
    REQUIRE(lease->validatePresentationPoint(point1).has_value());
    REQUIRE(lease->failPresentation(point1).has_value());

    REQUIRE(lease->acquire().has_value());
    const auto point2 = makePoint(rachis::QueueType::graphics, 41);
    auto firstOk = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    auto signalFailed = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(firstOk.has_value());
    REQUIRE(signalFailed.has_value());

    REQUIRE((*firstOk)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*firstOk)->markSubmitted().has_value());
    REQUIRE((*firstOk)->publishSubmissionPoint(point2).has_value());

    REQUIRE((*signalFailed)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*signalFailed)->markSubmitted().has_value());
    signalFailed->reset();

    auto blockedPresent = lease->validatePresentationPoint(point2);
    REQUIRE_FALSE(blockedPresent.has_value());
    CHECK(blockedPresent.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(lease->beginPresentation(point2).has_value());
    CHECK_FALSE(lease->cancel().has_value());
}

TEST_CASE("D3D12 swapchain lease rejects non-graphics queues, foreign devices, and regressing timelines cleanly",
          "[size:small][rhi][d3d12][swapchain][lease]")
{
    auto lease = std::make_shared<rachis::d3d12::SwapchainImageLeaseState>();

    auto unacquiredRef = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE_FALSE(unacquiredRef.has_value());
    CHECK(unacquiredRef.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(lease->acquire().has_value());

    auto computeRef = lease->retainCommandReference(lease, rachis::QueueType::compute);
    REQUIRE_FALSE(computeRef.has_value());
    CHECK(computeRef.error().code == rachis::ErrorCode::invalidArgument);

    auto copyRef = lease->retainCommandReference(lease, rachis::QueueType::copy);
    REQUIRE_FALSE(copyRef.has_value());
    CHECK(copyRef.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(lease->cancel().has_value());
    REQUIRE(lease->acquire().has_value());

    auto first = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(first.has_value());

    auto badComputeSubmit = (*first)->validateSubmission(rachis::QueueType::compute);
    REQUIRE_FALSE(badComputeSubmit.has_value());
    CHECK(badComputeSubmit.error().code == rachis::ErrorCode::invalidArgument);

    auto badCopySubmit = (*first)->validateSubmission(rachis::QueueType::copy);
    REQUIRE_FALSE(badCopySubmit.has_value());
    CHECK(badCopySubmit.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE((*first)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*first)->markSubmitted().has_value());
    CHECK_FALSE((*first)->markSubmitted().has_value());
    CHECK_FALSE((*first)->validateSubmission(rachis::QueueType::graphics).has_value());

    auto invalidPoint = (*first)->publishSubmissionPoint({});
    REQUIRE_FALSE(invalidPoint.has_value());
    CHECK(invalidPoint.error().code == rachis::ErrorCode::invalidArgument);

    auto computePoint = (*first)->publishSubmissionPoint(makePoint(rachis::QueueType::compute, 5, 1));
    REQUIRE_FALSE(computePoint.has_value());
    CHECK(computePoint.error().code == rachis::ErrorCode::invalidArgument);

    const auto validFirstPoint = makePoint(rachis::QueueType::graphics, 10, 1);
    REQUIRE((*first)->publishSubmissionPoint(validFirstPoint).has_value());
    CHECK_FALSE((*first)->publishSubmissionPoint(validFirstPoint).has_value());

    auto second = lease->retainCommandReference(lease, rachis::QueueType::graphics);
    REQUIRE(second.has_value());
    REQUIRE((*second)->validateSubmission(rachis::QueueType::graphics).has_value());
    REQUIRE((*second)->markSubmitted().has_value());

    auto foreignDevice = (*second)->publishSubmissionPoint(makePoint(rachis::QueueType::graphics, 11, 2));
    REQUIRE_FALSE(foreignDevice.has_value());
    CHECK(foreignDevice.error().code == rachis::ErrorCode::invalidArgument);

    auto regressed = (*second)->publishSubmissionPoint(makePoint(rachis::QueueType::graphics, 9, 1));
    REQUIRE_FALSE(regressed.has_value());
    CHECK(regressed.error().code == rachis::ErrorCode::invalidState);

    const auto validSecondPoint = makePoint(rachis::QueueType::graphics, 12, 1);
    REQUIRE((*second)->publishSubmissionPoint(validSecondPoint).has_value());
    REQUIRE(lease->beginPresentation(validSecondPoint).has_value());
}
