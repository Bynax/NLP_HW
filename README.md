### <center>NLP HOMEWORK</center>
- project structure:
    - I divide this project into 3 parts:**CW** for cut words, **POS** for Part-of-Speech tagging and **NER** for  Named Entity Recognition . Every part is combined by the following four files:
        - dataset:preprocess the dataset
        - model:build a model
    - evaluate: evaluate a model, calculate the accuracy,recall and F1_score
        - run: run the part
- CW
    - model:HMM
      - Parameter estimate:MLE
      - Decode:viterbi algorithm
      
    - Evaluate:
      - Accuracy: 0.7784937575513492
      - Recall: 0.7972449063763095
      - F1_score: 0.7877577634689054
      
    - model:HMM+AddOne Smoothing
    
    - Evaluate:
      - Accuracy:0.7720314111208059
      - Recall:0.802854079023344
      - F1_score:0.7871411241407198
      
    - Model:HMM+Smoothing 
    
      - Add-delta
        - 0.2 (0.7766468985414329, 0.7993895900354697, 0.7878541522702328)
        - 0.1 (0.7772249418184737, 0.798894663037202, 0.7879108363163034)
        - 0.005 (0.778851574454377, 0.7977398333745772, 0.7881825590872045)
        - 0.047 (0.7786173633440514, 0.7989771508702467, 0.7886658795749704)
        - 0.027(0.7794224117126538, 0.7992246143693805, 0.7891993157937607)
      - Most_Common:
        counter:[(0.0, 62686), (1.0, 424), (2.0, 203), (3.0, 127), (5.0, 90), (4.0, 82), (6.0, 79), (11.0, 49), (7.0, 48), (10.0, 46), (12.0, 45), (8.0, 43), (9.0, 39), (13.0, 37), (14.0, 36), (16.0, 31), (18.0, 26), (15.0, 25), (27.0, 22), (28.0, 21)]
      
      - Good-Turing
        - k=20 (0.7782969467493757, 0.796914955044131, 0.7874959243560482)
        - k=60 (0.7784402191427651, 0.7969974428771757, 0.7876095373955575)
        - k=100 (0.7784223672548546, 0.796914955044131, 0.7875601206488955)
      
    - test 上得分 (0.7961952159255097, 0.8043954261617062, 0.8002743152204608)
    