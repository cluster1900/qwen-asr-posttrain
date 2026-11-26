# 《从零到一：大模型底层原理、GPU 算子优化与任意模型后训练实战》

本目录包含面向普通开发者与高中学历学习者的完整技术书籍源文件、18 张原创高清矢量原理图、构建与自动审校脚本，以及面向 GitHub Release 的 GitHub Pages 在线阅读构建器。

> **整理说明**：为保持 Git `main` 分支轻量，编译产物 `book/dist/` 与静态站 `book/_site/` 已加入 `.gitignore`，正式成品统一归档发布在 GitHub Release **`book-v1.0.0`**（附 SHA-256 校验清单），并由 `.github/workflows/book-pages.yml` 直接从该 Release 拉取 HTML 成品发布至 GitHub Pages。

## GitHub Release 正式发布版（`book-v1.0.0`）

- Release 页面：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/book-v1.0.0`

| 资产文件名 | 格式 | 字节数 | SHA-256 |
| :--- | :---: | ---: | :--- |
| `zero-to-one-llm-posttraining-v1.0.0.html` | HTML 离线交互版 | `679,810` | `b73031d419cefcc52e6a3a132051ebb1be9f91f862225774c93b161c72ad2df8` |
| `zero-to-one-llm-posttraining-v1.0.0.pdf` | PDF 印刷版（83 页） | `10,442,936` | `b9f2a38d3f367e334e33a33e053efef2eaf46fbc9a5bdd7686e3ef1502d99814` |
| `zero-to-one-llm-posttraining-v1.0.0.epub` | EPUB3 电子书 | `885,215` | `30734bed7553825d9a9ff44d4582491ab8b79aeaa97f846ccd8dd928d403b80f` |
| `book-v1.0.0_SHA256SUMS.txt` | 校验清单 | `320` | `1e79f0fdba5a1f04b51a8b620e7c20fc1b1043905c292c60e17af5a4a3282ef9` |

## GitHub Pages 在线阅读（专为 Release HTML 打造）

- 自动部署工作流：[`.github/workflows/book-pages.yml`](../.github/workflows/book-pages.yml)
  - 每当发布/编辑 `book-v*` Release，或在 Actions 页面手动点击 `Run workflow` 时，自动通过 `gh release download` 拉取对应 Release 的 `.html` 成品与 `SHA256SUMS.txt`，校验 SHA-256 后运行 `book/scripts/build_pages.py` 生成静态站点（`index.html`、`404.html`、`release.json`、`.nojekyll`）并部署至 GitHub Pages。
  - 页面直接以完整交互阅读界面打开（侧边目录搜索、亮色/护眼/暗黑切换、字号调节），并在侧边栏内置 `PDF`、`EPUB3` 与 `Release` 一键下载入口。
- **本地一键从 Release 拉取并预览 GitHub Pages 站点**：

```bash
python3 book/scripts/build_pages.py --from-release book-v1.0.0 --serve 8080
# 浏览器打开 http://127.0.0.1:8080/
```

## 目录结构

```text
book/
├── README.md                              # 本说明文档
├── book.yaml                              # 书籍元数据与章节顺序清单
├── assets/
│   ├── cover.svg / cover.png              # 高清矢量封面与 EPUB3 封面图
│   └── diagrams/                          # 18 张原创高清矢量架构图/流程图/时序图/Roofline/显存图
├── chapters/                              # 14 篇 Markdown 章节源文（序言 + 12 章 + 3 个实战附录）
├── styles/
│   ├── book.css                           # HTML 交互版样式（含 Release 下载卡片）
│   ├── print.css                          # PDF 印刷版样式
│   └── epub.css                           # EPUB3 电子书样式
├── scripts/
│   ├── build_book.py                      # 一键同步编译 HTML + PDF + EPUB3 到 book/dist/
│   ├── audit_book.py                      # 自动审校（SVG/Python AST/数学账本/公式/三格式校验）
│   └── build_pages.py                     # 从 GitHub Release HTML 组装并预览 GitHub Pages 站点
├── dist/                                  # 本地构建输出（.gitignore 忽略，正式成品发布在 Release）
└── _site/                                 # 本地/CI 组装的 GitHub Pages 站点（.gitignore 忽略）
```

## 本地重建与自动审校

```bash
python3 book/scripts/build_book.py && python3 book/scripts/audit_book.py
```

构建依赖：`pandoc`（3.x+）与本机 Google Chrome（用于无头矢量 PDF 印刷与封面栅格化），不依赖外部网络 CDN。
