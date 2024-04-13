在本篇論文中，我們提出新穎的特徵擾動方式Mask Feature，它參考MAE提出的圖片遮罩方法，與多篇特徵擾動論文，使模型透過特徵遮罩來強化模型對鄰近特徵的關聯能力。同時，我們提出Student Consistency Loss，可以補足教師模型架構中學生的一致性正則化。此外，我們提出UDA領域包含半監督領域，可將UDA模型架構提供給半監督領域，也能將技巧共享，我們選用在UDA領域中經過多次改良的MIC作為本篇論文方法的基礎

綜合上述所提到的概念，本篇方法的貢獻如下:

1.本文提出Mask Feature，以patch為單位對特徵進行遮罩，使模型從有缺漏的特徵中預測出完整內容，強化模型對於特徵的關聯能力。

2.本文提出Student Consistency Loss，將consistency regulation加入教師模型架構，強化學生模型的強健性以提高整體準確率。

3.本文提出UDA領域包含半監督領域，將UDA領域中發展完善的MIC模型架構套用到半監督領域中，使發展完善的模型架構能跨領域處理問題。
![image](https://github.com/haha20331/MIC-Feature_mask/assets/67794071/208e2c52-cad5-4a37-8a88-6891e58fee12)
