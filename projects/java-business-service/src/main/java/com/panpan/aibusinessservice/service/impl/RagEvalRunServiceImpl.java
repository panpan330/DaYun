package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.dto.RagEvalRunView;
import com.panpan.aibusinessservice.dto.SaveRagEvalRunCommand;
import com.panpan.aibusinessservice.entity.RagEvalRun;
import com.panpan.aibusinessservice.mapper.RagEvalRunMapper;
import com.panpan.aibusinessservice.service.RagEvalRunService;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class RagEvalRunServiceImpl implements RagEvalRunService {

    private static final int MAX_LIST_LIMIT = 100;
    private static final int DEFAULT_LIST_LIMIT = 20;

    private final RagEvalRunMapper mapper;

    public RagEvalRunServiceImpl(RagEvalRunMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    @Transactional
    public void save(SaveRagEvalRunCommand command) {
        RagEvalRun run = new RagEvalRun();
        run.setRunId(command.runId());
        run.setRetriever(command.retriever());
        run.setStartedAt(command.startedAt());
        run.setCompletedAt(command.completedAt());
        run.setTopK(command.topK());
        run.setHitRate(command.hitRate());
        run.setRecall(command.recall());
        run.setPrecision(command.precision());
        run.setMrr(command.mrr());
        run.setCaseCount(command.caseCount());
        run.setDetailsJson(command.detailsJson());
        mapper.insert(run);
    }

    @Override
    public List<RagEvalRunView> listRecent(int limit) {
        int bounded = Math.min(Math.max(limit, 1), MAX_LIST_LIMIT);
        return mapper.listRecent(bounded).stream().map(RagEvalRunServiceImpl::toView).toList();
    }

    @Override
    public RagEvalRunView latestByRetriever(String retriever) {
        RagEvalRun run = mapper.latestByRetriever(retriever);
        return run == null ? null : toView(run);
    }

    private static RagEvalRunView toView(RagEvalRun run) {
        return new RagEvalRunView(
                run.getRunId(),
                run.getRetriever(),
                run.getStartedAt(),
                run.getCompletedAt(),
                run.getTopK(),
                run.getHitRate(),
                run.getRecall(),
                run.getPrecision(),
                run.getMrr(),
                run.getCaseCount(),
                run.getDetailsJson(),
                run.getCreatedAt()
        );
    }
}
