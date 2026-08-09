from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.rag.retriever import DEFAULT_TOP_K


DEFAULT_CACHE_TTL_SECONDS = 60.0
DEFAULT_CACHE_MAX_ENTRIES = 128
NEAR_TIMEOUT_RATIO = 0.8


class RagCacheKey(BaseModel):
    namespace: str = Field(min_length=1)
    digest: str = Field(min_length=16)
    components: dict[str, object] = Field(default_factory=dict)

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.digest}"


class RagCacheStats(BaseModel):
    hit_count: int = Field(ge=0)
    miss_count: int = Field(ge=0)
    set_count: int = Field(ge=0)
    evicted_count: int = Field(ge=0)
    current_entries: int = Field(ge=0)


@dataclass
class _CacheItem:
    value: Any
    created_at: float
    expires_at: float


class InMemoryTtlCache:
    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._items: dict[str, _CacheItem] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._set_count = 0
        self._evicted_count = 0

    def get(self, key: str) -> Any | None:
        normalized_key = _normalize_cache_key(key)
        now = self.clock()
        item = self._items.get(normalized_key)
        if item is None:
            self._miss_count += 1
            return None
        if item.expires_at <= now:
            del self._items[normalized_key]
            self._miss_count += 1
            self._evicted_count += 1
            return None

        self._hit_count += 1
        return item.value

    def set(self, key: str, value: Any) -> None:
        normalized_key = _normalize_cache_key(key)
        now = self.clock()
        self._delete_expired(now)
        if normalized_key not in self._items and len(self._items) >= self.max_entries:
            self._evict_oldest()

        self._items[normalized_key] = _CacheItem(
            value=value,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._set_count += 1

    def clear(self) -> None:
        self._items.clear()

    def stats(self) -> RagCacheStats:
        self._delete_expired(self.clock())
        return RagCacheStats(
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            set_count=self._set_count,
            evicted_count=self._evicted_count,
            current_entries=len(self._items),
        )

    def _delete_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, item in self._items.items()
            if item.expires_at <= now
        ]
        for key in expired_keys:
            del self._items[key]
            self._evicted_count += 1

    def _evict_oldest(self) -> None:
        oldest_key = min(
            self._items,
            key=lambda key: self._items[key].created_at,
        )
        del self._items[oldest_key]
        self._evicted_count += 1


class RagBatchPlan(BaseModel):
    item_count: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    batch_count: int = Field(ge=0)
    batches: list[list[str]] = Field(default_factory=list)


class RagOperationStage(str, Enum):
    EMBEDDING = "embedding"
    VECTOR_STORE = "vector_store"
    RERANK = "rerank"
    GENERATION = "generation"
    SECURITY = "security"


class RagOperationStatus(str, Enum):
    OK = "ok"
    NEAR_TIMEOUT = "near_timeout"
    TIMED_OUT = "timed_out"


class RagOperationTiming(BaseModel):
    stage: RagOperationStage
    elapsed_ms: float = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    status: RagOperationStatus


class RagDegradationMode(str, Enum):
    USE_CACHED_RETRIEVAL = "use_cached_retrieval"
    RETURN_SAFE_FALLBACK = "return_safe_fallback"
    RETURN_NO_CONTEXT = "return_no_context"


class RagDegradationDecision(BaseModel):
    stage: RagOperationStage
    mode: RagDegradationMode
    reason: str = Field(min_length=1)
    should_call_model: bool
    should_use_cache: bool
    user_message: str = Field(min_length=1)


RagPerformanceProtectionArea = Literal[
    "cache",
    "timeout",
    "degradation",
    "batching",
    "retrieval",
    "rerank",
    "generation",
]
RagPerformanceProtectionPriority = Literal["high", "medium", "low"]


class RagPerformanceProtectionRecommendation(BaseModel):
    area: RagPerformanceProtectionArea
    priority: RagPerformanceProtectionPriority
    reason: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    risk: str = Field(min_length=1)


class RagPerformanceProtectionReport(BaseModel):
    timing_count: int = Field(ge=0)
    timed_out_stages: list[RagOperationStage] = Field(default_factory=list)
    near_timeout_stages: list[RagOperationStage] = Field(default_factory=list)
    cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    degradation_mode: RagDegradationMode | None = None
    recommendation_count: int = Field(ge=0)
    high_priority_count: int = Field(ge=0)
    recommendations: list[RagPerformanceProtectionRecommendation] = Field(
        default_factory=list
    )


def build_retrieval_cache_key(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = None,
    permission_group: str | None = None,
    business_domain: str | None = None,
    doc_type: str | None = None,
    source: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    collection_name: str | None = None,
    namespace: str = "rag_retrieval",
) -> RagCacheKey:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")
    _validate_positive_int(top_k, field_name="top_k")
    _validate_score_threshold(score_threshold)
    if embedding_dimension is not None:
        _validate_positive_int(embedding_dimension, field_name="embedding_dimension")

    components: dict[str, object] = {
        "query_hash": _hash_text(normalized_query),
        "top_k": top_k,
        "score_threshold": score_threshold,
        "permission_group": _normalize_optional_text(permission_group),
        "business_domain": _normalize_optional_text(business_domain),
        "doc_type": _normalize_optional_text(doc_type),
        "source": _normalize_optional_text(source),
        "embedding_model": _normalize_optional_text(embedding_model),
        "embedding_dimension": embedding_dimension,
        "collection_name": _normalize_optional_text(collection_name),
    }
    serialized = json.dumps(
        components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return RagCacheKey(
        namespace=_normalize_namespace(namespace),
        digest=_hash_text(serialized),
        components=components,
    )


def build_batch_plan(
    items: Sequence[str],
    *,
    batch_size: int,
) -> RagBatchPlan:
    _validate_positive_int(batch_size, field_name="batch_size")
    normalized_items = [item for item in items]
    for item in normalized_items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("batch item must be a non-blank string")

    batches = [
        normalized_items[index : index + batch_size]
        for index in range(0, len(normalized_items), batch_size)
    ]
    return RagBatchPlan(
        item_count=len(normalized_items),
        batch_size=batch_size,
        batch_count=len(batches),
        batches=batches,
    )


def assess_operation_timing(
    stage: RagOperationStage,
    *,
    elapsed_ms: float,
    timeout_seconds: float,
    near_timeout_ratio: float = NEAR_TIMEOUT_RATIO,
) -> RagOperationTiming:
    if elapsed_ms < 0:
        raise ValueError("elapsed_ms must be greater than or equal to 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    if not 0 < near_timeout_ratio < 1:
        raise ValueError("near_timeout_ratio must be between 0 and 1")

    timeout_ms = timeout_seconds * 1000
    if elapsed_ms >= timeout_ms:
        status = RagOperationStatus.TIMED_OUT
    elif elapsed_ms >= timeout_ms * near_timeout_ratio:
        status = RagOperationStatus.NEAR_TIMEOUT
    else:
        status = RagOperationStatus.OK

    return RagOperationTiming(
        stage=stage,
        elapsed_ms=elapsed_ms,
        timeout_seconds=timeout_seconds,
        status=status,
    )


def choose_degradation_decision(
    stage: RagOperationStage,
    *,
    has_cached_retrieval: bool,
    has_safe_chunks: bool = False,
) -> RagDegradationDecision:
    if has_cached_retrieval:
        return RagDegradationDecision(
            stage=stage,
            mode=RagDegradationMode.USE_CACHED_RETRIEVAL,
            reason="当前链路失败，但存在可用的检索缓存。",
            should_call_model=True,
            should_use_cache=True,
            user_message="正在使用最近一次可用的知识库检索结果生成回答。",
        )

    if has_safe_chunks:
        return RagDegradationDecision(
            stage=stage,
            mode=RagDegradationMode.RETURN_SAFE_FALLBACK,
            reason="模型生成或后续步骤不可用，但仍有安全检索资料可用于兜底。",
            should_call_model=False,
            should_use_cache=False,
            user_message="当前模型回答暂时不可用，但已检索到相关资料，请稍后重试或查看来源资料。",
        )

    return RagDegradationDecision(
        stage=stage,
        mode=RagDegradationMode.RETURN_NO_CONTEXT,
        reason="当前链路失败，且没有缓存或安全资料可用于兜底。",
        should_call_model=False,
        should_use_cache=False,
        user_message="当前知识库服务暂时不可用，无法根据知识库回答这个问题。",
    )


def build_rag_performance_protection_report(
    *,
    timings: Sequence[RagOperationTiming] = (),
    cache_stats: RagCacheStats | None = None,
    degradation_decision: RagDegradationDecision | None = None,
) -> RagPerformanceProtectionReport:
    timed_out_stages = [
        timing.stage
        for timing in timings
        if timing.status is RagOperationStatus.TIMED_OUT
    ]
    near_timeout_stages = [
        timing.stage
        for timing in timings
        if timing.status is RagOperationStatus.NEAR_TIMEOUT
    ]
    cache_hit_rate = _cache_hit_rate(cache_stats) if cache_stats else None
    recommendations: list[RagPerformanceProtectionRecommendation] = []

    _append_timing_recommendations(recommendations, timings)
    if cache_stats is not None:
        _append_cache_recommendations(
            recommendations,
            cache_stats=cache_stats,
            cache_hit_rate=cache_hit_rate,
        )
    if degradation_decision is not None:
        _append_degradation_recommendations(recommendations, degradation_decision)

    deduped_recommendations = _dedupe_recommendations(recommendations)
    return RagPerformanceProtectionReport(
        timing_count=len(timings),
        timed_out_stages=timed_out_stages,
        near_timeout_stages=near_timeout_stages,
        cache_hit_rate=cache_hit_rate,
        degradation_mode=(
            degradation_decision.mode if degradation_decision is not None else None
        ),
        recommendation_count=len(deduped_recommendations),
        high_priority_count=sum(
            1 for recommendation in deduped_recommendations if recommendation.priority == "high"
        ),
        recommendations=deduped_recommendations,
    )


def format_rag_performance_protection_report(
    report: RagPerformanceProtectionReport,
) -> list[str]:
    cache_hit_rate = (
        "-" if report.cache_hit_rate is None else f"{report.cache_hit_rate:.4f}"
    )
    lines = [
        "RAG performance protection report",
        f"timings: {report.timing_count}",
        f"timed_out_stages: {_format_stage_list(report.timed_out_stages)}",
        f"near_timeout_stages: {_format_stage_list(report.near_timeout_stages)}",
        f"cache_hit_rate: {cache_hit_rate}",
        (
            "degradation_mode: "
            f"{report.degradation_mode.value if report.degradation_mode else '-'}"
        ),
        f"recommendations: {report.recommendation_count}",
        f"high_priority: {report.high_priority_count}",
    ]
    for recommendation in report.recommendations:
        lines.append(
            f"- {recommendation.priority} {recommendation.area}: "
            f"{recommendation.reason} evidence={recommendation.evidence} "
            f"action={recommendation.suggested_action} risk={recommendation.risk}"
        )
    return lines


def _normalize_cache_key(key: str) -> str:
    normalized = key.strip()
    if not normalized:
        raise ValueError("cache key must not be blank")
    return normalized


def _normalize_namespace(namespace: str) -> str:
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace must not be blank")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_positive_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_score_threshold(score_threshold: float | None) -> None:
    if score_threshold is None:
        return
    if not isinstance(score_threshold, int | float) or isinstance(score_threshold, bool):
        raise ValueError("score_threshold must be a number")
    if score_threshold < 0:
        raise ValueError("score_threshold must be greater than or equal to 0")


def _append_timing_recommendations(
    recommendations: list[RagPerformanceProtectionRecommendation],
    timings: Sequence[RagOperationTiming],
) -> None:
    for timing in timings:
        if timing.status is RagOperationStatus.TIMED_OUT:
            recommendations.append(
                _performance_recommendation(
                    area="timeout",
                    priority="high",
                    reason=f"{timing.stage.value} reached its timeout budget.",
                    evidence=(
                        f"elapsed_ms={timing.elapsed_ms:.2f}, "
                        f"timeout_seconds={timing.timeout_seconds:.2f}"
                    ),
                    suggested_action=(
                        "Set an explicit timeout, record the failure, and route "
                        "the request to a degradation path."
                    ),
                    risk="Timeouts that are too strict can reject valid slow requests.",
                )
            )
            recommendations.append(
                _performance_recommendation(
                    area="degradation",
                    priority="high",
                    reason="A timed-out stage needs a defined fallback behavior.",
                    evidence=f"stage={timing.stage.value}",
                    suggested_action=(
                        "Prefer cached retrieval when safe, otherwise return a "
                        "safe fallback or no-context response."
                    ),
                    risk="Fallback responses may be less complete than normal RAG answers.",
                )
            )
        elif timing.status is RagOperationStatus.NEAR_TIMEOUT:
            area: RagPerformanceProtectionArea = (
                "rerank"
                if timing.stage is RagOperationStage.RERANK
                else "generation"
                if timing.stage is RagOperationStage.GENERATION
                else "retrieval"
            )
            recommendations.append(
                _performance_recommendation(
                    area=area,
                    priority="medium",
                    reason=f"{timing.stage.value} is close to its timeout budget.",
                    evidence=(
                        f"elapsed_ms={timing.elapsed_ms:.2f}, "
                        f"timeout_seconds={timing.timeout_seconds:.2f}"
                    ),
                    suggested_action=(
                        "Review candidate count, model/provider latency, and "
                        "whether this stage needs batching or fallback."
                    ),
                    risk="Optimizing latency by reducing candidates can hurt answer quality.",
                )
            )


def _append_cache_recommendations(
    recommendations: list[RagPerformanceProtectionRecommendation],
    *,
    cache_stats: RagCacheStats,
    cache_hit_rate: float | None,
) -> None:
    request_count = cache_stats.hit_count + cache_stats.miss_count
    if request_count == 0:
        return
    if cache_hit_rate is not None and cache_hit_rate < 0.3:
        recommendations.append(
            _performance_recommendation(
                area="cache",
                priority="medium",
                reason="Retrieval cache hit rate is low.",
                evidence=f"cache_hit_rate={cache_hit_rate:.4f}",
                suggested_action=(
                    "Review cache TTL, cache key components, query normalization, "
                    "and whether repeated read-only requests are cacheable."
                ),
                risk="Over-broad caching can leak scoped data or serve stale knowledge.",
            )
        )
    if cache_stats.evicted_count > cache_stats.set_count:
        recommendations.append(
            _performance_recommendation(
                area="cache",
                priority="low",
                reason="Cache eviction count is higher than set count.",
                evidence=(
                    f"evicted={cache_stats.evicted_count}, set={cache_stats.set_count}"
                ),
                suggested_action="Review cache max_entries and TTL settings.",
                risk="Increasing cache size raises memory usage.",
            )
        )


def _append_degradation_recommendations(
    recommendations: list[RagPerformanceProtectionRecommendation],
    decision: RagDegradationDecision,
) -> None:
    if decision.mode is RagDegradationMode.USE_CACHED_RETRIEVAL:
        recommendations.append(
            _performance_recommendation(
                area="cache",
                priority="high",
                reason="Current degradation path depends on cached retrieval.",
                evidence=f"stage={decision.stage.value}",
                suggested_action=(
                    "Ensure cache keys include tenant, permission scope, retrieval "
                    "parameters, embedding model, and collection version."
                ),
                risk="Unsafe cache keys can leak data across users or tenants.",
            )
        )
    elif decision.mode is RagDegradationMode.RETURN_SAFE_FALLBACK:
        recommendations.append(
            _performance_recommendation(
                area="degradation",
                priority="medium",
                reason="The system can return safe retrieved evidence without model generation.",
                evidence=f"stage={decision.stage.value}",
                suggested_action=(
                    "Make the fallback response explicit and avoid presenting it as "
                    "a full model-generated answer."
                ),
                risk="Users may need a follow-up when only evidence is returned.",
            )
        )
    else:
        recommendations.append(
            _performance_recommendation(
                area="degradation",
                priority="high",
                reason="The system has no cache or safe chunks available for fallback.",
                evidence=f"stage={decision.stage.value}",
                suggested_action=(
                    "Return a clear no-context/service-unavailable message and log "
                    "the failed stage for observability."
                ),
                risk="Frequent no-context fallbacks can reduce user trust.",
            )
        )


def _cache_hit_rate(stats: RagCacheStats) -> float | None:
    request_count = stats.hit_count + stats.miss_count
    if request_count == 0:
        return None
    return round(stats.hit_count / request_count, 6)


def _performance_recommendation(
    *,
    area: RagPerformanceProtectionArea,
    priority: RagPerformanceProtectionPriority,
    reason: str,
    evidence: str,
    suggested_action: str,
    risk: str,
) -> RagPerformanceProtectionRecommendation:
    return RagPerformanceProtectionRecommendation(
        area=area,
        priority=priority,
        reason=reason,
        evidence=evidence,
        suggested_action=suggested_action,
        risk=risk,
    )


def _dedupe_recommendations(
    recommendations: Sequence[RagPerformanceProtectionRecommendation],
) -> list[RagPerformanceProtectionRecommendation]:
    deduped: list[RagPerformanceProtectionRecommendation] = []
    seen: set[tuple[str, str]] = set()
    for recommendation in recommendations:
        key = (recommendation.area, recommendation.reason)
        if key in seen:
            continue
        deduped.append(recommendation)
        seen.add(key)
    return deduped


def _format_stage_list(stages: Sequence[RagOperationStage]) -> str:
    if not stages:
        return "-"
    return ", ".join(stage.value for stage in stages)
