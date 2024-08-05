在本篇論文中，我們提出了一種觀點，即無監督域適應（Unsupervised Domain Adaptation, UDA）領域包含了半監督學習（Semi-Supervised Learning, SSL）領域。基於此，我們提出了一種基於UDA的SSL方法，即UDA-Based SSL，並將多種在UDA領域中的方法引用到本文的方法架構中。同時，我們提出了一種學生一致性框架（Student Consistency Framework），通過將一致性正則化引入教師-學生模型架構中，以強化模型的強健性。此外，我們提出了一種新穎的特徵擾動方式——特徵遮罩（Feature Masking），該方法參考了 \cite{MAE,mic}提出的圖像遮罩方法，以及多篇特徵擾動論文 \cite{psmt,unimatch}，使模型通過遮罩特徵來強化其對鄰近特徵的關聯能力。

綜合上述所提到的概念，本篇方法的貢獻如下:
\begin{itemize}
\item UDA-Based SSL方法：本文提出了一種基於UDA的SSL方法，將兩個領域的方法概念進行融合，為現有技術提供了新的方向，從而提升性能並擴展應用範圍。

\item 學生一致性框架（Student Consistency Framework）：本文提出了一種通過一致性正則化強化學生模型架構的方法，使學生模型和教師模型的強健性提升，以提高整體準確率。

\item 特徵遮罩（Feature Masking）方法：本文提出了一種以Patch為單位對特徵進行遮罩的方法，使模型能從有缺漏的特徵中預測出完整內容，從而強化模型對於特徵的關聯能力。

\item 實驗結果顯示卓越性能：實驗結果顯示，我們的方法在Cityscapes資料集標記比例1/2、 1/4 和1/8 ，Classic Pascal VOC 資料集標記比例Full、 1/2 和1/4 ，Blended Pascal VOC 資料集標記比例1/2、 1/4和1/8 上均取得了最先進（SOTA）的成績。此外，在消融實驗中，我們驗證了每個提出方法的有效性。
![full_framework](https://github.com/user-attachments/assets/39222f65-fd8b-477f-9c5e-6a953372cc35)
