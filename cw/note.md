### <center>HMM中文分词</center>
- 分词评价标准
   - 准确率P
    切分结果中正确的分词数/切分结果中所有的分词数
   - 召回率R
    切分结果中正确的分词数/标准答案中的分词数
   - F-score
    P和R的调和平均值     
- 参数学习
    暂定使用最大似然估计进行参数的学习
- 分词结果
    将上一步的分词结果利用维特比算法计算最佳路径得出