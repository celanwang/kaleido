# 从原文到可冻结 token

原始文件和来源哈希保留，处理时移除头尾声明与目录，按实际章节标题划分文档。
完整章去重后，对至少80字符的规范化段落全局去重；按书分层，SHA排序固定40/4/4章划分。
跨划分5词shingle Jaccard审计只排查近乎相同章节，不能证明没有短语或语义重叠。
这是同书章节留出，不是未见作者或未见书籍泛化。

ByteBPE从256个字节加EOS开始，按训练文本regex块的加权相邻频次合并；同频按token对排序。
不跨regex块/章节合并，encode按最早合并rank执行；未知Unicode落到UTF-8字节，完整文本可逆。
EOS只能显式添加，文本里的特殊标记不会被自动解释。单个生成token可能不是有效UTF-8，
展示生成文本时显式用replacement解码，训练原文roundtrip仍严格检查。
词表目标512不是保证：数据不足会提前停止。格式保存pattern、顺序合并、EOS和版本，并严格校验。
这不是tiktoken兼容实现，也不把不同tokenizer的困惑度直接比较。

运行：`uv run --locked --all-extras pytest tests/test_bpe_corpus.py`；
完整demo在 `python -m kaleido.demos.language_model`。tokenizer只见train，文件源指纹和逐章token数保存。
