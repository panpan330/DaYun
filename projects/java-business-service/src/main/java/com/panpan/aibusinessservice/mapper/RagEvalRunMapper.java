package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.RagEvalRun;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface RagEvalRunMapper {

    int insert(RagEvalRun run);

    List<RagEvalRun> listRecent(@Param("limit") int limit);

    RagEvalRun latestByRetriever(@Param("retriever") String retriever);
}
