# 《从零到一：大模型底层原理、GPU 算子优化与任意模型后训练实战》

本目录包含面向普通开发者与高中学历学习者的完整技术书籍源文件、18 张原创高清矢量原理图、自动审校工具，以及同步生成的 **HTML / PDF / EPUB3** 三格式电子书成品。

## 成品阅读入口（`dist/`）

| 格式 | 文件路径 | 适用阅读方式 |
| :--- | :--- | :--- |
| **HTML（离线单文件）** | `dist/从零到一大模型底层与后训练实战.html` | 浏览器直接双击打开；支持侧边目录搜索、亮色/护眼/暗黑切换、字号缩放、离线矢量图与 MathML 公式 |
| **PDF（高清印刷版）** | `dist/从零到一大模型底层与后训练实战.pdf` | 桌面/平板/打印阅读；含高清矢量封面、完整目录、A4 页眉页码、高清晰度矢量图与公式 |
| **EPUB3（标准电子书）** | `dist/从零到一大模型底层与后训练实战.epub` | Apple Books、Calibre、Thorium、Kobo、Kindle、微信读书等全平台电子书阅读器 |

## 目录结构

```text
book/
├── README.md                              # 本说明文档
├── book.yaml                              # 书籍元数据与章节顺序清单
├── assets/
│   ├── cover.svg / cover.png              # 高清矢量封面与 EPUB3 封面图
│   └── diagrams/                          # 18 张原创高清矢量架构图/流程图/时序图/Roofline/显存图
├── chapters/                              # 14 个 Markdown 章节源文（序言 + 12 章 + 3 个实战附录）
├── styles/
│   ├── book.css                           # HTML 交互版样式
│   ├── print.css                          # PDF 印刷版样式
│   └── epub.css                           # EPUB3 电子书样式
├── scripts/
│   ├── build_book.py                      # 一键同步编译 HTML + PDF + EPUB3
│   └── audit_book.py                      # 自动审校（SVG/Python AST/数学账本/公式/三格式校验）
└── dist/                                  # 最终构建产物（HTML / PDF / EPUB3）
```

## 一键重新构建与自动审校

修改任意章节或图表后，在仓库根目录执行：

```bash
python3 book/scripts/build_book.py && python3 book/scripts/audit_book.py
```

构建依赖：`pandoc`（3.x+）与本机 Google Chrome（用于无头矢量 PDF 印刷与封面栅格化），不依赖外部网络 CDN。
