Dataset-1 appendix copy of Part 4.

The prompt, context, hard-case selector, safety gate, Qwen model and inference settings are intentionally unchanged from the final D2/D3 Part-4 code. Only dataset routing, hybrid source path and output namespace are changed.

Run after Dataset-1 model training and Part-3 hybrid:
    python run_part4.py --stage test --dataset dataset1
    python report_part4_performance.py

Outputs are isolated under results/dataset1_appendix/part4_llm/.
