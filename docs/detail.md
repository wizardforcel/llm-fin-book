# 《大模型金融应用：从基本面理解到可信部署》书籍细纲

## 设计说明

本书设**六章**：基本面量化、因子挖掘、信号预测、风控、智能客服、知识库。每节按四要素展开：**（1）标题；（2）一段话简介；（3）可用的 MCP / Skill；（4）实操路径**。

工具选择遵循一个原则：**MCP 负责取数与执行，Skill 负责约束推理路径**。当前金融 MCP 生态中，量化投研类工具（Qlib、FactorMiner、wraquant）相对成熟；风控与客服类工具更偏参考实现或企业内建，适合在受控环境中做能力验证；知识库类则以 RAG 基础设施组件为主。


## 第一章 基本面量化

### 1.1 大模型重构基本面信息处理链条

**（2）简介**：传统基本面量化受限于财报滞后、文本非结构化、分析师覆盖不均。大模型的能力边界需要被精确划定——它擅长抽取、比较和跨期一致性检查，但不擅长精确计算。标准流水线应为：文档解析→实体与指标抽取→口径对齐→时序面板构建→信号生成，财务计算交给确定性工具。

**（3）可用 MCP / Skill**：
- **DataSinking MCP**：6 个工具，支持美中日韩台财报全文的章节级获取（`list sections` / `fetch one section`），token 友好，专为 RAG 和 LLM 阅读设计。
- **Octagon MCP**：覆盖 SEC filings、earnings call transcripts（10 年历史）、财务指标与比率分析，自然语言问询返回带引用回答。
- **Anthropic 金融服务 agent 模板**：`earnings-reviewer` 技能可读取 transcript 和 filings、更新模型并标记对投资逻辑有实质影响的变化；`model-builder` 从 filings 和数据源构建财务模型。

**（4）实操**：先用 DataSinking 的 `get_section` 只取财报的“管理层讨论与分析”章节（而非全文），将 section 文本喂给 `earnings-reviewer` Skill 提取关键指标和 guidance 变化；财务精确计算（同比、比率）通过外部 Python 或确定性 MCP 工具完成，LLM 只负责口径对齐和项目名称映射。风险点：DataSinking 返回 Markdown，需检查表格列对齐是否在转换中丢失。

### 1.2 财报与公告的深度解析

**（2）简介**：三表勾稽的模型辅助应定位为“提示异常线索”，规则引擎完成校验；MD&A 的语调、不确定性表述和风险提示密度可作为时序因子；公告事件识别需要类型化和强度打分，且必须控制文本发布时间戳以防止前视偏差。

**（3）可用 MCP / Skill**：
- **sec-edgar-mcp / edgartools**：将 10-K、10-Q、8-K、Form 4 解析为类型化数据和结构化 XBRL，避免 LLM 从原始 HTML 中“猜”数字。
- **DataSinking MCP**：章节级获取财报全文，保留文档层级结构，适合定位“风险因素”“重大事项”等特定章节。
- **`backtesting-bias-avoidance` Skill**：其 point-in-time 检查逻辑可复用到公告文本的发布时间戳校验，确保信号在公告实际可得时点之后才进入面板。

**（4）实操**：用 sec-edgar-mcp 拉取某公司过去 8 个季度 10-Q 的 XBRL 数据，让 LLM 对比营收、应收账款、经营现金流三项的跨期变化，输出“需要人工复核的科目与方向”列表，再交规则引擎校验勾稽关系。公告事件识别用 8-K 和 Form 4 数据做类型初筛；**不要**让模型直接判断“是否造假”。

### 1.3 研报与调研纪要的批量理解

**（2）简介**：分析师预期修正和调研纪要是基本面文本信号的重要来源。情绪的非对称性需要被显式建模——负面情绪对下行风险的预警价值显著高于正面情绪对上涨的指示价值。多源文本冲突时的置信度分配规则应由人工预设优先级，而非依赖模型“记住”。

**（3）可用 MCP / Skill**：
- **Octagon MCP**：earnings call transcript 覆盖 10 年历史和 8000+ 上市公司，适合批量语义处理。
- **DataSinking MCP**：支持中美日韩台财报，可用于研报中引用的财务数据核验。
- **Anthropic `earnings-reviewer` Skill**：读取 transcript 和 filings，提取 guidance 变化和 thesis-relevant 变化。

**（4）实操**：用 Octagon 拉取某公司最近 4 个季度 transcript，要求 Skill 逐季度输出结构化字段（确定性表述比例、风险词频、前瞻性指引的具体数字）；按时间对齐后计算情绪变化的一阶差分作为周度因子原料。负面情绪加权高于正面的处理**在因子构建脚本中实现**，而非依赖模型生成时的隐式判断。

### 1.4 基本面策略构建与边界

**（2）简介**：大模型文本因子需要与传统基本面因子做互补性检验，并强制通过回测过拟合门禁。基本面逻辑的可解释性是刚性要求——信号必须能映射回具体经营事实。大模型在财务精确计算和会计政策变更识别中存在系统性弱点，必须由规则库与人工复核兜底。

**（3）可用 MCP / Skill**：
- **Qlib MCP**：`qlib_factor_analysis` 做 IC 分析和分位数收益，`qlib_backtest_topk` 做 TopK 回测返回年化/夏普/最大回撤，`qlib_get_data` 按 Qlib 表达式查询价格与因子数据。
- **wraquant-mcp**：暴露 1097 个量化金融函数为 MCP 工具，包括 `risk.sharpe_ratio`、`vol.garch_fit`、`regimes.detect_regimes`，可做因子衰减的滚动监测和风险调整评估。
- **`skill-backtest-overfit`**：计算 Deflated Sharpe Ratio 和 PBO，返回 PASS/FAIL，因子上线前的强制统计门禁。

**（4）实操**：在 Qlib MCP 中注册“调研纪要情绪因子”或“MD&A 不确定性因子”为自定义字段，调用 `qlib_factor_analysis` 做 IC 衰减曲线；样本外快速衰减时调用 `skill-backtest-overfit`。**Qlib MCP 必须先调用 `qlib_init`**，A 股日线数据约 500MB，首次需 `qlib_download_data` 预留下载时间。


## 第二章 因子挖掘

### 2.1 从人工假设到 LLM 引导搜索

**（2）简介**：大模型在因子挖掘中的价值是扩展假设空间和提升逻辑可解释性，而非直接“预测涨跌”。三种技术路线值得关注：公式化 Alpha 的 LLM 引导搜索（MCTS + 回测反馈 + 频繁子树规避）、多智能体辩论生成因子、以及“因子改进而非从零生成”。判断因子好坏的权力必须交给回测统计和人工评审，不能由提出因子的同一模型兼任。

**（3）可用 MCP / Skill**：
- **FactorMiner Skill**：模块化技能架构，封装完整的因子挖掘闭环。`mine`（Ralph loop）是论文忠实实现（检索、生成、评估、准入、进化记忆）；`helix` 是增强超集，加入 do-calculus 因果验证、regime 条件评估、多专家辩论生成和 SymPy 规范化。
- **FactorMiner MCP 替代方案**：当 FactorMiner MCP server 连接时，`mine_factors` 和 `helix_mine` 以工具形式暴露同一工作流，返回结构化 session 摘要。
- **Qlib MCP**：`qlib_expression_help` 提供算子列表和 Alpha158 说明，`qlib_factor_analysis` 直接验证生成的 Qlib 表达式。

**（4）实操**：先用 FactorMiner Skill 的 Ralph loop 在已验证数据集上生成候选因子库（示例：`factorminer -o output/run1 mine --data market_data.csv --iterations 40 --batch-size 16 --target 30`），再对存活因子调用 Qlib MCP 的 `qlib_factor_analysis` 做独立 IC 验证。需要因果稳健性时切换到 `helix --causal --regime --debate --canonicalize`，但 Phase 2 特征有额外计算成本。

### 2.2 非结构化数据的因子化 Pipeline

**（2）简介**：从文本源到可检验因子的转换链包括实体识别、关系抽取、情绪打分、时序聚合和因子标准化。情绪 Alpha 的非对称性应正面与负面分别建模。文本信号与量价信号在时间尺度和信噪比上不匹配，融合规则应分层（文本信号作为门控条件），而非简单线性加权。

**（3）可用 MCP / Skill**：
- **StockTwits MCP / xueqiu-mcp**：散户情绪和社交量动量的原始数据源。
- **`skill-factor-alpha191-alpha101`**：从 OHLCV CSV 原生计算 Alpha101、Alpha191 和 Qlib 兼容 Alpha158，生成宽表因子矩阵。
- **`CTAAgents/quant-skills`**：遗传编程做因子发现，可作文本情绪与量价因子融合时的交互项搜索。

**（4）实操**：跑通最小管道——StockTwits 或 xueqiu 取股票池周度情绪得分 → 按时间对齐价格面板 → `skill-factor-alpha191-alpha101` 生成量价基线因子 → 情绪因子与量价因子做相关性矩阵，检查文本信号是否提供增量信息。融合规则用分层：文本信号只在量价信号处于特定状态（如低波动或趋势确认）时才激活，避免简单加权掩盖信号冲突。

### 2.3 因子评估、复现与失效管理

**（2）简介**：LLM 生成因子面临两个特有陷阱：多重检验偏误（大量生成候选必然产生幸存者）和输出随机性导致的不可复现。每个因子必须记录生成时的完整 prompt 和模型版本，固定随机种子。因子被市场广泛认知后需要触发退役机制。

**（3）可用 MCP / Skill**：
- **wraquant-mcp**：提供 `regimes.detect_regimes` 和完整的风险/波动率分析工具链，可用于因子衰减的 regime 条件化监测——判断因子失效是否与市场状态切换相关。
- **`skill-backtest-overfit`**：DSR 和 PBO 计算，给出 PASS/FAIL。
- **`agent-backtest-lab`**：专为 LLM 交易 Agent 设计的统计审计工具，防止前视泄漏和 PBO 过拟合。
- **`alpha-monitor`（来自 `alpha-skills`）** ：因子衰减监测，IC 持续低于阈值触发退役信号。

**（4）实操**：每个 LLM 生成的因子强制走“回测 → DSR/PBO 统计门禁 → 持续监测”三段流程。**必须在因子定义中记录完整 prompt、模型版本和随机种子**，否则同一 prompt 两次生成的表达式可能不同，无法复现。对 IC 超过阈值的因子，再用独立样本外区间验证，避免“在同一数据集上既搜索又验证”的循环偏误。


## 第三章 信号预测

### 3.1 金融时序基础模型与领域错位

**（2）简介**：通用时序基础模型（TSFM）在低信噪比、强非平稳的金融数据上存在结构性“领域错位”。判断时序模型是否可用，关键不是看它跑赢了多少，而是看它在**哪些市场状态下系统性落后于朴素基线**——这些状态应被 regime detection 门控覆盖。

**（3）可用 MCP / Skill**：
- **AdvancedFinancialML MCP Server**：LSTM-GNN 混合模型做股价预测、动态波动率建模（GARCH/EGARCH/TGARCH）、市场 regime 检测，提供 `market_regime_detector` 和 `predict_stock_lstm_gnn_adaptive` 等工具。
- **wraquant-mcp**：`regimes.detect_regimes` 和 `vol.garch_fit` 可作为独立的 regime 基线，与 ML 模型的 regime 输出交叉验证。
- **Role-Based MCP Framework（学术）** ：多智能体编排 + MCP 集成实时数据做金融预测，可作为对照架构参考。

**（4）实操**：用 AdvancedFinancialML 的 `predict_stock_lstm_gnn_adaptive` 跑基线预测，同时用 `market_regime_detector` 输出 regime 概率序列；将 LSTM 预测误差按 regime 分组统计，识别“模型在哪些 regime 下显著劣于朴素动量”。**regime 检测的输出直接作为模型静音/激活的门控信号**，而非让模型自己判断“现在该不该信自己”。

### 3.2 多模态信息融合的信号增强

**（2）简介**：文本信号与价格序列的融合需要分层架构：文本质量过滤 → 二元门控 → 时序模型输出。滚动微调（参数微调 + 推理参数调整并行）能使系统适应市场状态的结构性变化，但收益归因需要区分“模型适配”和“文本过滤”各自的贡献。信号聚合与仓位映射必须分离——模型输出状态判断，风控规则决定仓位动作。

**（3）可用 MCP / Skill**：
- **StockTwits MCP / xueqiu-mcp**：文本情绪源，送入预测模型前先用 LLM 做质量过滤（去 spam、低信息量帖子）。
- **AdvancedFinancialML MCP**：`predict_volatility_comprehensive` 提供波动率预测，`generate_trading_signals_advanced` 生成带仓位 sizing 和止损止盈的交易信号——**注意**：应将信号生成与仓位映射分离使用，模型侧只用预测和信号方向，仓位逻辑由独立规则层执行。
- **wraquant-mcp 的 prompt templates**：217 个引导式多步工作流模板，可用于组织“质量过滤→门控→预测”的分层调用序列。

**（4）实操**：分层管道——第一层 LLM 做文本质量过滤和情绪打分；第二层规则或浅层模型将文本信号转为“是否激活”的二元门控；第三层时序模型（如 AdvancedFinancialML 的 LSTM-GNN）在门控激活时输出信号。滚动微调通过 `alpha-monitor` 或独立的监测输出触发再训练 pipeline，模型参数微调与推理参数调整作为两个可独立回退的环节。

### 3.3 信号预测的可信部署

**（2）简介**：择时信号的可解释性要求高于选股——单笔择时错误的代价集中且即时。信号置信度不能替代责任归属。大模型择时在制度性断点（政策突变、交易机制变更）面前的脆弱性，要求人工否决机制是编排层的硬规则，而非模型判断的一部分。

**（3）可用 MCP / Skill**：
- **wraquant-mcp**：`regimes.detect_regimes` 可用于标记“制度性断点”或异常 regime，作为人工否决的触发信号。
- **人工否决的工程化**：没有现成 MCP 能“替代”人工否决，需要在编排层设计硬规则——当 regime detector 输出预设的“断点”标记时，无论模型置信度多高，信号进入 `pending_human_review` 队列，不自动生成仓位动作。
- **AdvancedFinancialML 的 `simulate_portfolio_scenarios`** ：Monte Carlo 情景分析（VaR、CVaR、stress test），可用于评估信号在极端情景下的表现边界。

**（4）实操**：将人工否决机制实现为编排层的**状态机**：`regime_detector` 返回“制度性断点”标记 → 信号强制进入 `pending_human_review` → 人工确认后才能继续。**不要**让 LLM 在生成信号时“考虑是否应该人工复核”——这个决策必须由确定性规则做出。


## 第四章 风控

### 4.1 风控场景的大模型能力匹配矩阵

**（2）简介**：风控是大模型金融应用中容错空间最小的场景。大模型的角色是“增强推理与线索发现”，而非“替代规则引擎与审批责任”。信贷风控中的语义提取、反欺诈中的逻辑链检测、AML 中的文档审查辅助，是当前能力匹配度较高的方向。多模态反欺诈（假章、合成语音）需要专用小模型，不宜由通用 LLM 直接判断。

**（3）可用 MCP / Skill**：
- **RAG_DB_UBS（Bank Surveillance Console）** ：只读 RAG + MCP 工具包，覆盖 AML/fraud/credit/KYC。LLM 通过 MCP 写**只读 SQL** 查询结构化表（22 张表，含 customers、transactions、loans 等），同时用文档 RAG 检索约 12k 真实银行文本。**四层只读安全模型**：语句验证（仅单条 SELECT）、READ ONLY 事务、8 秒超时、行数上限 200。
- **Anthropic `kyc-screener` agent 模板**：内置 `kyc-rules` 技能，将机构的 KYC/AML 规则应用于解析后的 onboarding 记录，打包升级事项供合规审查。

**（4）实操**：用 RAG_DB_UBS 的 `run_sql` 模式，让 LLM 生成 SQL 但经四层守卫执行。**最佳实践**：在数据库层额外给 MCP 使用的 DB 用户只授予 SELECT 权限，使应用层守卫成为第二道防线而非唯一防线。“可疑交易报告辅助生成”设计为：LLM 检索相关政策和历史案例（RAG）→ 生成结构化草稿 → 人工审核，模型不直接提交报告。

### 4.2 RAG 作为风控合规基础设施

**（2）简介**：风控决策对“依据可追溯”的刚性要求，使 RAG 从“增强生成质量”上升为合规基础设施。关键不是“检索到相关段落”，而是**检索到正确的否定条件和例外条款**。结构化查数（SQL）与非结构化查依据（文档 RAG）的混合检索是风控 RAG 的基本形态。

**（3）可用 MCP / Skill**：
- **RAG_DB_UBS 的文档 RAG 层**：约 12k 银行文本的语义检索，与结构化 SQL 查询并存。**关键设计**：关系型数据库用 SQL 查询，不用向量“搜索”——“和数据库对话”意味着 LLM 通过 MCP 写 SQL，向量 RAG 用于非结构化文本。
- **findocs-mcp**：eval-first 的 MCP 服务器，提供 `search_docs`、`answer_question`（带引用或**拒绝回答**）、`ingest_doc`。其 eval harness 在 retrieval recall、ranking quality、answer faithfulness 和 **refusal correctness** 上做 CI 回归门禁——“一个自信的错误答案比诚实的‘未找到’更糟”。

**（4）实操**：风控 RAG 应增加“条款完整性检查”：监管制度类文档的检索结果必须覆盖相关条款的完整章节，而非只命中关键词片段。`findocs-mcp` 的 `answer_question` 自带 confidence gate——当检索置信度低时返回“not found”而非编造答案，这一机制应被保留，**不要**在上层用提示词绕过 refusal gate。

### 4.3 可解释性与责任边界

**（2）简介**：大模型输出在风控流程中的定位是“建议”，而非“决定”。可解释性的场景化要求：对客户提供结果解释，对监管提供逻辑链路，对内部审计提供决策轨迹。信贷审批的最终责任始终在人，模型不能成为责任主体。

**（3）可用 MCP / Skill**：
- **RAG_DB_UBS 的 explainable detectors**：8 个规则化检测器返回结构化发现（触发规则、涉及字段、阈值），并展示**精确的 SQL 和 flagged rows**，而非 LLM 的自由文本解释。
- **`finance-skills`**：防止 Agent “从记忆中发明”财务数字，强制走确定性计算路径。同样思路迁移到风控：凡涉及金额、比率、阈值的输出，必须来自 MCP 工具调用而非模型生成。
- **人工核验队列**：在 MCP 工具返回结构中强制包含 `confidence`、`sources`、`requires_human_review` 字段，编排层据此决定是否自动推进。

**（4）实操**：风控 Agent 的每条输出必须携带 `source`（政策条款 ID 或交易记录 ID）、`reasoning_trace`（调用了哪些工具、检索了哪些文档）和 `requires_human_review`（布尔值）。`requires_human_review` **由规则生成**（如涉及金额超过阈值、检测器置信度低于阈值、命中多个冲突规则），而非模型自评。


## 第五章 智能客服

### 5.1 金融客服的特殊约束与技术架构

**（2）简介**：金融智能客服的核心约束不是“像人一样说话”，而是答案权威性、口径统一性和监管留痕义务。技术栈应分层：语音识别 → 意图理解 → 知识检索（RAG）→ 话术生成 → 合规校验。意图分类、知识检索和话术生成不应由同一个通用 LLM 同时承担——路由层需要确定性。

**（3）可用 MCP / Skill**：
- **YugabyteDB 语音金融 Chatbot 架构**：MCP server 作为桥梁，暴露 `GetLatestTransactions(userID)` 等工具，LLM 通过 MCP 调用 SQL 查询和 pgvector 语义搜索，再生成回答。语音侧 STT/TTS 完成转换，MCP 负责“取数 + 取知识”。
- **pgvector + HNSW 方案**：800+ 合作银行的 RAG 平台实践显示，调优后的 pgvector HNSW 索引可在多租户负载下保持稳定延迟，并通过 MCP 暴露聊天导出接口。
- **Anthropic 金融服务 agent 模板**：包含 meeting preparer、portfolio analysis、compliance checks 等技能，可作为客服中“账户/产品知识”环节的参考架构。

**（4）实操**：最小可行架构参照 YugabyteDB 模式——pgvector 做 FAQ 和产品信息的语义索引，MCP 暴露结构化查询工具（如 `GetLatestTransactions`），Node.js 后端作为 MCP client 组装 system prompt 和用户元数据。金融客服的特殊约束需在 MCP 工具返回中**强制携带来源文档 ID 和版本号**，话术生成后经规则层校验（关键词黑名单、数字一致性）才输出。

### 5.2 “事前—事中—事后”服务闭环

**（2）简介**：事前知识运营自动化（从历史对话提取未覆盖问题和过期条目）、事中实时坐席辅助（RAG 推荐话术，延迟需控制在亚秒级）、事后全量智能质检（从人工抽检转向 LLM 全量扫描）构成服务闭环。实时环节的延迟预算决定了模型选择和缓存策略，离线质检可以使用较大模型。

**（3）可用 MCP / Skill**：
- **Retail Bank Ops Copilot 参考**：处理 40% 的一线客户咨询，基于核心银行上下文，通过 MCP 连接核心银行和 CRM 并做字段级访问控制，具有人工升级和**完整审计日志**用于投诉处理。
- **FinClaw 话术合规审查 Skill（`bank-t226-compliance-operations-talk-track-review-assistant`）** ：对营销/客服话术做合规审核，识别违规表述、缺失风险提示或夸大收益，输出修订建议与升级路径。另有 `bank-t169-retail-finance-complaint-comfort-assistant` 用于客诉安抚沟通话术的合规边界生成。

**（4）实操**：事前阶段用历史对话 + 知识库文档做差异比对，LLM 提取“未覆盖问题”和“过期条目”，更新决策由知识运营人员做出。事中阶段实时坐席辅助用轻量嵌入 + 缓存控制亚秒级延迟；话术生成后先经合规审查 Skill 过一遍再推送给坐席。事后质检可离线批量跑，用 FinClaw 的话术审查 Skill 对全量通话转写做三类违规扫描，输出带时间戳和片段的报告。

### 5.3 智能客服的风险与治理

**（2）简介**：大模型“自由发挥”与金融口径刚性之间存在结构性冲突。多轮对话中的合规风险会累积——单轮合规不等于全流程合规。投诉处理、复杂纠纷、涉及法律后果的对话必须保留人工接管通道，不能用模型置信度替代责任归属。

**（3）可用 MCP / Skill**：
- **规则过滤器 MCP**：在 MCP 工具层和生成层之间插入必经关卡，检查模型输出是否包含禁用表述、是否引用已废止产品条款、数字是否与结构化查询结果一致，通过后才放行。
- **FinClaw 话术合规审查 Skill**：可同时用于“生成前”的话术模板审查和“生成后”的输出质检。
- **人工接管触发器**：在编排层维护对话级 `compliance_context`，每轮 MCP 工具调用读写该上下文。触发词（“投诉”“起诉”“监管”“曝光”）或连续两轮未解决 → 立即转人工，**硬规则不进入模型自主决策分支**。

**（4）实操**：部署规则过滤器 MCP 作为所有客服输出的必经关卡；将人工接管触发器实现为编排层状态机，与模型置信度判断**解耦**。多轮合规检查在每轮生成后执行，而不是只在会话结束时检查——合规风险是累积的，事后检查无法阻止中间轮次的违规输出已经触达客户。


## 第六章 知识库

### 6.1 金融知识的特殊性对 RAG 架构的约束

**（2）简介**：金融知识库不是“把文档塞进向量数据库”，而是以检索精度、时效性和可审计性为核心指标的知识工程体系。通用 RAG 在金融场景的失效模式包括：切片破坏条款语义完整性、检索遗漏否定条件、生成环节引入原文不存在的推论。金融 RAG 的评估必须纳入“引用可追溯率”“口径一致性”和“时效合规性”。

**（3）可用 MCP / Skill**：
- **DataSinking MCP**：`list sections` / `fetch one section` 按章节获取财报全文，天然保留文档层级结构，适合金融文档的层级化切分。
- **findocs-mcp**：eval-first 设计，eval harness 在 retrieval recall、ranking quality、answer faithfulness 和 refusal correctness 上做 CI 回归门禁，基线回归则 build 失败。
- **PIT-aware RAG**：检索时强制 `publish_date <= as_of_date`，避免历史研究检索到未来公告；所有文档必须携带 `publish_date` 元数据。

**（4）实操**：金融 RAG 的第一个工程决策是**切片策略**，不是选向量库。以监管制度为例，按“章—节—条款”做层级切分，每个 chunk 携带 `doc_id`、`chapter`、`article`、`effective_date`、`superseded_by` 元数据。检索时**先做元数据预过滤**（只检索 `effective_date <= decision_date` 且 `superseded_by` 为空的条款），再做向量检索。通用固定长度切片会破坏条款语义完整性，在金融场景中不可接受。

### 6.2 知识切片与检索的金融化改造

**（2）简介**：层级化切分以条款/章节为语义边界，保留金融文档的层级编号体系。预过滤（正则或元数据）在检索前缩小候选范围，提升效率与准确性。混合检索（BM25 + 稠密检索）和金融术语引导的嵌入模型微调，是应对金融术语特殊性的必要手段。

**（3）可用 MCP / Skill**：
- **findocs-mcp**：Postgres + pgvector（HNSW，cosine）+ 本地 MiniLM 嵌入，默认零 API 成本、无需密钥，eval gate 在 CI 中可复现。工具包括 `search_docs`、`answer_question`（confidence gate + citations 或 refusal）、`ingest_doc`。
- **finance_data_llm**：混合检索（dense + BM25）+ reranking，支持 SEC filings 的嵌入和搜索。
- **multi-agent-trading 的 Qdrant + bge-reranker 架构**：Docker Compose 一键启动 agent-api + finance-mcp + qdrant，可作为检索层可复现基线。

**（4）实操**：先用 findocs-mcp 或 finance_data_llm 搭建最小检索链路，重点调试**否定条件和例外条款的召回**。构造测试查询如“某产品在什么情况下**不**适用冷静期”，检查检索结果是否覆盖条款中的“除外”和“但书”部分。通用嵌入模型在金融术语上表现不佳时，用金融术语词典引导嵌入模型微调或加入同义词扩展层。**预过滤的正则应由金融术语词典生成**，不由模型临时编写。

### 6.3 知识库作为金融 Agent 的“事实底座”

**（2）简介**：知识库是前五章场景共同的事实底座。路由逻辑的质量决定 Agent 的可靠性——何时查知识库、何时查数据库、何时触发人工。系统必须能够识别并上报“知识空白”和“规则冲突”状态，生成“无可靠依据”的拒答，而非编造一个看似合理的答案。

**（3）可用 MCP / Skill**：
- **RAG_DB_UBS 的混合架构**：同时暴露只读 SQL 工具和文档 RAG 工具，Agent 根据问题类型选择“查数”还是“查文档”，是“知识库 + 业务系统”路由逻辑的可运行参考。
- **findocs-mcp 的 refusal gate**：`answer_question` 在检索置信度低时返回“not found”，这一机制应保留为知识库的**默认行为**，而非例外路径。
- **路由决策 Skill**：输入用户问题和上下文，输出 `{knowledge_base | database | human | no_answer}`。关键规则：相关性低于阈值，或检索到的条款存在 `superseded_by` 冲突且无法自动消解 → 必须输出 `no_answer` 或 `human`。

**（4）实操**：先在一个受限域（如票据规则或某一产品线）内跑通“知识库 + 路由 + 拒答”完整链路，再扩展到多业务线。每一条 Agent 输出必须携带至少一个 `source_chunk_id`，且该 chunk 的 `effective_date` 必须在决策时点之前。**PIT 检查不应只在检索时做一次**——在生成输出的最终校验环节再检查一次 source chunk 的时点有效性，防止中间步骤引入过期信息。


## 结语：六章场景的共同工程门槛

六个业务版块的 MCP/Skill 选型可以不同，但可信部署的工程门槛是共通的：**硬规则**（人工否决、合规过滤器、read-only 守卫）、**统计门禁**（DSR/PBO、CI eval gate）和**责任可追溯**（source ID、reasoning trace、requires_human_review）必须在架构层被设计为不可绕过的约束，而非提示词层面的“建议”。Skill 和 MCP 解决的是“模型能调用什么”，不解决“模型应该在什么约束下决策”——后者比工具选型更能决定系统是否可信。