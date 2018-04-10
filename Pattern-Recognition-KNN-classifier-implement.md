# Pattern Recognition - KNN classifier implement

  

 

1. Store [UCI dataset - Wine Data Set](https://archive.ics.uci.edu/ml/datasets/wine) as pandas dataframe, and store every feature and label into variable X and y respectively.
```    
    import pandas as pd
    
    def load_feature(filename):
        df = pd.read_csv(filename)
        feature = df.drop('Class',1)
        return feature
    
    def load_label(filename):
        df = pd.read_csv(filename)
        label = df.Class    
        return label
    
    def main():
        X = load_feature('wine.data')
        y = load_label('wine.data')
```
![print(X.head())](https://d2mxuefqeaa7sj.cloudfront.net/s_B6396EEFC5C76E03FBA1C4BD007CC4555B362DE3F3C76AEF56335C3A30B0C786_1523279485937_image.png)
![print(y.head())](https://d2mxuefqeaa7sj.cloudfront.net/s_B6396EEFC5C76E03FBA1C4BD007CC4555B362DE3F3C76AEF56335C3A30B0C786_1523279655694_2018-04-09+21-12-35+.png)



2. Split training data and testing data.(based on sklearn.model_selection.train_test_split)
  The store them in X_train, X_test, y_train, y_test.
    ```
    from sklearn.model_selection import train_test_split
    def main():
        X = load_feature('wine.data')
        y = load_label('wine.data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    ```



3. Calculate cosine similarity of all testing data and training data, then store them in the cs_array(row:X_test_arr, col: X_train).
    ```
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    def cos_sim(X_train, X_test, y_train, y_test):
        X_train_arr = np.array(X_train.values)
        X_test_arr = np.array(X_test.values)
        cs_array = cosine_similarity(X_test_arr,X_train_arr)
        return cs_array
    ```
![print(cs_array)](https://d2mxuefqeaa7sj.cloudfront.net/s_B6396EEFC5C76E03FBA1C4BD007CC4555B362DE3F3C76AEF56335C3A30B0C786_1523279809296_image.png)




4. **knn_classify**  
  1. choose k largest values of cosine similarity in each row. (It means that k training data nodes which is close to the current testing data node.) 
  2. Compare labels of this k nodes and choose the most common label as the result of the prediction of the testing data
    ```
    def knn_classify(X_train, X_test, y_train, y_test, k):
        cs_array = cos_sim(X_train, X_test, y_train, y_test)
        k_list = []
        y_pred_list = []    
        for i in range(len(cs_array)):
            k_list = heapq.nlargest(k, range(len(cs_array[i])), cs_array[i].take) 
            #k_list stores index of k largest cosine similarity value
            class_list = []
            for idx in k_list:
                class_list.append(y_train.iloc[idx])
                #class_list stores k prediction classes of each node.
            a = np.array(class_list)
            counts = np.bincount(a)
            print(np.argmax(counts))    
            #choose the most common class as the result prediction
            y_pred_list.append(np.argmax(counts))
            print("====================")
        y_pred = pd.Series(y_pred_list)
        return y_pred
    ```
![](https://d2mxuefqeaa7sj.cloudfront.net/s_B6396EEFC5C76E03FBA1C4BD007CC4555B362DE3F3C76AEF56335C3A30B0C786_1523280363388_image.png)



5. Check the classifier report and accuracy 
    ```
    from sklearn.metrics import classification_report as clf_report
    from sklearn.metrics import accuracy_score
    
    y_pred = knn_classify(X_train, X_test, y_train, y_test, 5)
    accurancy = accuracy_score(y_test,y_pred)
    report = clf_report(y_test,y_pred)
    ```
![](https://d2mxuefqeaa7sj.cloudfront.net/s_B6396EEFC5C76E03FBA1C4BD007CC4555B362DE3F3C76AEF56335C3A30B0C786_1523280431669_image.png)


The code is on: https://github.com/SnoopyKnight/knn_classifier

## Reference:
- http://enginebai.logdown.com/posts/241676/knn
- https://pandas.pydata.org/pandas-docs/stable/
- http://scikit-learn.org/stable/
- http://www.numpy.org/

