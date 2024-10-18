# OceanBase AI 动手实战营

[英文版](./README.md)

## 介绍

在这个动手实战营中，我们会构建一个 RAG 聊天机器人，用来回答与 OceanBase 文档相关的问题。它将采用开源的 OceanBase 文档仓库作为多模数据源，把文档转换为向量和结构化数据存储在 OceanBase 中。用户提问时，该机器人将用户的问题作为输入在数据库中进行文档检索，结合文档检索的结果，借助智谱 AI 提供的大语言模型能力来回答用户的问题。

该机器人将由以下几个组件组成：

1. 将文档转换为向量的文本嵌入模型，BGE-M3
2. 提供存储和查询文档向量和其他结构化数据的数据库，OceanBase
3. 若干分析用户问题、基于检索到的文档和用户问题生成回答的 LLM 智能体，利用智谱 AI 的大模型能力
4. 与用户交互的聊天界面，采用 Streamlit 搭建

![RAG 流程](./demo/rag-flow.png)

## 准备工作

注意：如果您正在参加 OceanBase AI 动手实战营，您可以跳过以下步骤 1 ~ 3。所有所需的软件都已经在机器上准备好了。:)

1. 安装 [Python 3.9+](https://www.python.org/downloads/) 和 [pip](https://pip.pypa.io/en/stable/installation/)

2. 安装 [Poetry](https://python-poetry.org/docs/)

```bash
python3 -m pip install poetry
```

3. 安装 [Docker](https://docs.docker.com/engine/install/)

4. 注册 [智谱 AI](https://open.bigmodel.cn/) 账号并获取 API Key

![智谱 AI](./demo/zhipu-dashboard.png)

![智谱 API Key](./demo/zhipu-api-key.png)

## 构建聊天机器人

### 1. 部署 OceanBase 集群

#### 1.1 启动 OceanBase docker 容器

您可以使用以下命令启动一个 OceanBase docker 容器：

```bash
docker run --ulimit stack=4294967296 --name=ob433 -e MODE=mini -e OB_MEMORY_LIMIT=8G -e OB_DATAFILE_SIZE=10G -p 127.0.0.1:2881:2881 -d quay.io/oceanbase/oceanbase-ce:4.3.3.0-100000142024101215
```

如果上述命令执行成功，将会打印容器 ID，如下所示：

```bash
af5b32e79dc2a862b5574d05a18c1b240dc5923f04435a0e0ec41d70d91a20ee
```

#### 1.2 检查 OceanBase 初始化是否完成

容器启动后，您可以使用以下命令检查 OceanBase 数据库初始化状态：

```bash
docker logs -f ob433
```

初始化过程大约需要 2 ~ 3 分钟。当您看到以下消息（底部的 `boot success!` 是必须的）时，说明 OceanBase 数据库初始化完成：

```bash
cluster scenario: express_oltp
Start observer ok
observer program health check ok
Connect to observer ok
Initialize oceanbase-ce ok
Wait for observer init ok
+----------------------------------------------+
|                 oceanbase-ce                 |
+------------+---------+------+-------+--------+
| ip         | version | port | zone  | status |
+------------+---------+------+-------+--------+
| 172.17.0.2 | 4.3.3.0 | 2881 | zone1 | ACTIVE |
+------------+---------+------+-------+--------+
obclient -h172.17.0.2 -P2881 -uroot -Doceanbase -A

cluster unique id: c17ea619-5a3e-5656-be07-00022aa5b154-19298807cfb-00030304

obcluster running
Trace ID: 08f99c98-8c37-11ef-ad07-0242ac110002
If you want to view detailed obd logs, please run: obd display-trace 08f99c98-8c37-11ef-ad07-0242ac110002
Get local repositories and plugins ok
Open ssh connection ok
Connect to observer ok
Create tenant test ok
Exec oceanbase-ce-4.3.3.0-100000142024101215.el8-3eee13839888800065c13ffc5cd7c3e6b12cb55c import_time_zone_info.py ok
Exec oceanbase-ce-4.3.3.0-100000142024101215.el8-3eee13839888800065c13ffc5cd7c3e6b12cb55c import_srs_data.py ok
obclient -h172.17.0.2 -P2881 -uroot@test -Doceanbase -A

optimize tenant with scenario: express_oltp ok
Trace ID: 3c50193c-8c37-11ef-ace2-0242ac110002
If you want to view detailed obd logs, please run: obd display-trace 3c50193c-8c37-11ef-ace2-0242ac110002
check tenant connectable
tenant is connectable
boot success!
```

使用 `Ctrl + C` 退出日志查看界面。

#### 1.3 测试数据库部署情况（可选）

可以使用 mysql 客户端连接到 OceanBase 集群，检查数据库部署情况。

```bash
mysql -h127.0.0.1 -P2881 -uroot@test -A -e "show databases"
```

如果部署成功，您将看到以下输出：

```bash
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| oceanbase          |
| test               |
+--------------------+
```

### 2. 安装依赖

我们使用 Poetry 来管理聊天机器人项目的依赖项。您可以使用以下命令安装依赖项：

```bash
poetry install
```

如果您正在使用动手实战营提供的机器，您会看到以下消息，因为依赖都已经预先安装在了机器上：

```bash
Installing dependencies from lock file

No dependencies to install or update
```

### 3. 设置环境变量

我们准备了一个 `.env.example` 文件，其中包含了聊天机器人所需的环境变量。您可以将 `.env.example` 文件复制到 `.env` 并更新 `.env` 文件中的值。

```bash
cp .env.example .env
# 更新 .env 文件中的值，特别是 API_KEY 和数据库连接信息
vi .env
```

`.env.example` 文件的内容如下，如果您正在按照动手实战营的步骤进行操作（使用智谱 AI 提供的 LLM 能力），您只需要更新 `API_KEY` 为您从智谱 AI 控制台获取的值，其他值可以保留为默认值。

```bash
API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # 把这个配置项更新为您从智谱 AI 上获取到的 API_KEY
LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL="glm-4-flash"

HF_ENDPOINT=https://hf-mirror.com
BGE_MODEL_PATH=BAAI/bge-m3

DB_HOST="127.0.0.1"
DB_PORT="2881"
DB_USER="root@test"
DB_NAME="test"
DB_PASSWORD=""
```

### 4. 准备 BGE-M3 模型

BGE-M3 是一个预训练模型，可以将文本转换为向量。它在多种语言的嵌入任务中表现良好，可以用于将 OceanBase 的开源文档嵌入成为向量。

我们通过执行以下命令准备 BGE-M3 模型：

```bash
poetry run python utils/prepare_bgem3.py
```

在实战训练营提供的机器上模型已经下载完成。如果模型已经下载完成，这一步大约需要半分钟来加载模型。当模型准备好时，您将看到以下消息：

```bash
Fetching 30 files: 100%|████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 104509.24it/s]
/root/.cache/pypoetry/virtualenvs/ai-workshop-aLQYZfdO-py3.10/lib/python3.10/site-packages/FlagEmbedding/BGE_M3/modeling.py:335: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  colbert_state_dict = torch.load(os.path.join(model_dir, 'colbert_linear.pt'), map_location='cpu')
/root/.cache/pypoetry/virtualenvs/ai-workshop-aLQYZfdO-py3.10/lib/python3.10/site-packages/FlagEmbedding/BGE_M3/modeling.py:336: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')

===================================
BGEM3FlagModel loaded successfully！
===================================
```

### 5. 准备文档数据

#### 5.1 下载预处理数据并加载（速度快）

在这一步中，我们将加载预处理的文档数据到 OceanBase 数据库中。

```bash
# 加载预处理的文档数据
poetry run python utils/load.py --source_file ~/data.json
```

加载数据大约需要 2 分钟。您将看到以下输出：（SAWarnings 可以忽略）

```bash
args Namespace(table_name='corpus', source_file='/root/data.json', skip_create=False, insert_batch=100)
  0%|                                                                                                                                                                                                                                                          | 0/27412 [00:00<?, ?it/s]
/root/.cache/pypoetry/virtualenvs/ai-workshop-aLQYZfdO-py3.10/lib/python3.10/site-packages/pyobvector/client/ob_vec_client.py:329: SAWarning: Unknown schema content: '  VECTOR KEY `vidx` (`embedding`) WITH (DISTANCE=L2,M=16,EF_CONSTRUCTION=256,LIB=VSAG,TYPE=HNSW, EF_SEARCH=64) BLOCK_SIZE 16384'
  table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
100%|██████████████████████████████████████████████████████████████████| 27412/27412 [01:44<00:00, 262.02it/s]
```

注意：`data.json` 文件是一个预处理的文档数据文件，包含文档的向量数据和元数据。它是由步骤 5.2 处理得到、并通过执行 `utils/extract.py` 脚本提取出来的。如果您不想使用预处理数据而是打算自己嵌入文档并插入数据库，请转到步骤 5.2。

#### 5.2 克隆文档仓库并处理（速度慢）

注意：步骤 5.2 在 CPU 机器上通常要花费几个小时甚至更长的时间。

在该步骤中，我们将克隆 OceanBase 开源文档仓库并处理它们，生成文档的向量数据和其他结构化数据后将数据插入到我们在步骤 1 部署好的 OceanBase 数据库中。

```bash
cd doc_repos
git clone --single-branch --branch V4.3.3 https://github.com/oceanbase/oceanbase-doc.git
git clone --single-branch --branch V4.3.0 https://github.com/oceanbase/ocp-doc.git
git clone --single-branch --branch V4.3.1 https://github.com/oceanbase/odc-doc.git
git clone --single-branch --branch V4.2.5 https://github.com/oceanbase/oms-doc.git
git clone --single-branch --branch V2.10.0 https://github.com/oceanbase/obd-doc.git
git clone --single-branch --branch V4.3.0 https://github.com/oceanbase/oceanbase-proxy-doc.git
cd ..
```

```bash
# 把文档的标题转换为标准的 markdown 格式
poetry run python convert_headings.py \
  doc_repos/oceanbase-doc/zh-CN \
  doc_repos/ocp-doc/zh-CN \
  doc_repos/odc-doc/zh-CN \
  doc_repos/oms-doc/zh-CN \
  doc_repos/obd-doc/zh-CN \
  doc_repos/oceanbase-proxy-doc/zh-CN

# 生成文档向量和元数据
poetry run python embed_docs.py --doc_base doc_repos/oceanbase-doc/zh-CN
poetry run python embed_docs.py --doc_base doc_repos/ocp-doc/zh-CN --component ocp
poetry run python embed_docs.py --doc_base doc_repos/odc-doc/zh-CN --component odc
poetry run python embed_docs.py --doc_base doc_repos/oms-doc/zh-CN --component oms
poetry run python embed_docs.py --doc_base doc_repos/obd-doc/zh-CN --component obd
poetry run python embed_docs.py --doc_base doc_repos/oceanbase-proxy-doc/zh-CN --component odp
```

如果你想要将上述步骤生成并插入到数据库中的数据提取出来保存在 `my-data.json` 的文件中，可以执行以下命令：

```bash
poetry run python utils/extract.py --output_file ~/my-data.json
```

这就是我们获取预处理数据 `data.json` 的方法。

### 6. 启动聊天界面

执行以下命令启动聊天界面：

```bash
poetry run streamlit run --server.runOnSave false chat_ui.py
```

访问终端中显示的 URL 来打开聊天机器人应用界面。

```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.xxx.xxx.xxx:8501
  External URL: http://xxx.xxx.xxx.xxx:8501 # 这是您可以从浏览器访问的 URL
```

![](./demo/chatbot-ui.png)

## FAQ

### 1. 如何更改用于生成回答的 LLM 模型？

您可以通过更新 `.env` 文件中的 `LLM_MODEL` 环境变量来更改 LLM 模型。默认值是 `glm-4-flash`，这是智谱 AI 提供的免费模型。还有其他可用的模型，如 `glm-4-air`、`glm-4-plus`、`glm-4-long` 等。您可以在[智谱 AI 网站](https://open.bigmodel.cn) 上找到完整的模型列表。

### 2. 是否可以在初始加载后更新文档数据？

当然可以。您可以通过运行 `embed_docs.py` 脚本插入新的文档数据。例如：

```bash
# 这将在当前目录中嵌入所有 markdown 文件，其中包含 README.md 和 LEGAL.md
poetry run python embed_docs.py --doc_base .

# 或者您可以指定要插入数据的表
poetry run python embed_docs.py --doc_base . --table_name my_table
```

然后您可以在启动聊天界面之前指定 `TABLE_NAME` 环境变量，来指定聊天机器人将查询哪张表：

```bash
TABLE_NAME=my_table poetry run streamlit run --server.runOnSave false chat_ui.py
```

### 3. 如何知道嵌入和检索过程中数据库执行的操作？

当您自己插入文档时，可以设置 `--echo` 标志来查看脚本执行的 SQL 语句，如下所示：

```bash
poetry run python embed_docs.py --doc_base . --table_name my_table --echo
```

您将看到以下输出：

```bash
2024-10-16 03:17:13,439 INFO sqlalchemy.engine.Engine
CREATE TABLE my_table (
        id VARCHAR(4096) NOT NULL,
        embedding VECTOR(1024),
        document LONGTEXT,
        metadata JSON,
        component_code INTEGER NOT NULL,
        PRIMARY KEY (id, component_code)
)
...
```

您还可以在启动聊天界面之前设置 `ECHO=true`，以查看聊天界面执行的 SQL 语句。

```bash
ECHO=true TABLE_NAME=my_table poetry run streamlit run --server.runOnSave false chat_ui.py
```

## 附录

### 图像搜索应用

凭借 OceanBase 的向量存储和检索能力，我们还可以构建一个图像搜索应用。该应用将把图像嵌入到向量中并存储在数据库中。用户可以上传图像，应用程序将搜索并返回数据库中最相似的图像。

注意：您需要自己准备一些图像并将 `Image Base` 配置更新到打开的 UI 中。如果您本地没有可用的图像，可以在线下载数据集，例如在 Kaggle 上与 Animals 有关的数据集。

```bash
# 安装依赖
poetry install

# 启动图像搜索应用界面
poetry run streamlit run --server.runOnSave false image_search_ui.py
```
