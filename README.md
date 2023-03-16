# CS221 - Natural Language Processing

## Đề tài: Gán nhãn từ loại Tiếng việt sử dụng mô hình Hidden Markov và thuật toán Viterbi:

Gồm có 2 bài toán nhỏ là:

1. Bài toán tách từ
2. Gán nhãn từ loại

## Xây dựng bộ ngữ liệu

Bộ ngữ liệu bao gồm 62 câu được tổng hợp từ nhiều nguồn. Bộ ngữ liệu sau đó được được tách từ và gán nhãn thủ công dựa vào [bộ quy tắc của VLPS](https://vlsp.hpda.vn/demo/vcl/PoSTag.htm).

## Tách từ:

Chạy file: `longest_matching.ipynb`

Tách từ bằng phương pháp Longest Matching so sánh với thư viện VnCoreNLP.

|                | Longest Matching | VnCoreNLP |
| -------------: | ---------------: | --------: |
|       Accuracy |          0.90625 |  0.915761 |
|      Precision |         0.950893 |  0.943723 |
|         Recall |         0.835294 |  0.854902 |
|             F1 |         0.889353 |  0.897119 |
|  True Positive |              213 |       218 |
| False Positive |               11 |        13 |
|     Total True |              667 |       674 |
|   Total Errors |               97 |        79 |

## Gán nhãn từ loại

Chạy file: `pos_tagging.ipynb`

Bộ dữ liệu sẽ được chia thành 2 tập train và test với tỉ lệ 80:20. Thực hiện huấn luyện mô hình HMM trên tập train sau đó gán nhãn từ loại bằng phương pháp HMM+Vieterbi và đem so sánh độ chính xác với thu viện VnCoreNLP.

|                              | Accuracy |
| ---------------------------- | -------- |
| HMM + Viterbi trên tập train | 78%      |
| HMM + Viterbi trên tập test  | 57%      |
| VnCoreNLP trên tập train     | 88%      |
| VnCoreNLP trên tập test      | 93%      |
