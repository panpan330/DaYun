package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.dto.RagEvalRunView;
import com.panpan.aibusinessservice.dto.SaveRagEvalRunCommand;
import java.util.List;

public interface RagEvalRunService {

    void save(SaveRagEvalRunCommand command);

    List<RagEvalRunView> listRecent(int limit);

    RagEvalRunView latestByRetriever(String retriever);
}
