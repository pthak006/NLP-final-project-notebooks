# NLP-final-project-notebooks

## Details of all the files 

1. **bert-base-cased.py**: BERT base cased model python code written to run the model on the cluster.

2. **bert-base-cased-output.png**: output screenshot of the running of the bert-base-cased model.

3. **xlnet.py**: XLNet base cased model python code written to run the model on the cluster.

4. **XLNet-base-cased**.jpg: Output of the XLNet-base-cased model run on the cluster. 

5. **bert-base-stemming.py**: bert base cased model with stemming python code written to run the model on the cluster.

6. **output_bert-based-stemming.txt**: output txt file of the bert base cased model with stemming run on the cluster.

7. **bert-base-lemmatization.py**: bert base cased model with lemmatization python code written to run the model on the cluster.

8. **bert-base-lemmatization.txt**: output txt file of the bert base cased model with lemmatization run on the cluster.

9. **xlnet-stemming.py**: xlnet base cased model with stemming python code written to run the model on the cluster.

10. **xlnet-stemming-output.txt**: output txt file of the xlnet base cased model with stemming run on the cluster.

11. **bert-base-uncased.py**: bert base uncased model python code written to run the model on the cluster.

12. **output_bert_uncased.txt**: output txt file of the bert basse uncased model run on the cluster.

13. **BERT_large_uncased.ipynb**: ipynb file of the bert large uncased model run on google colab notebook. Could not run the code on the cluster as memory specification was exceeding with 16 batch size (cannot change the batch size because we have kept the initial setting of all the experiments same).

14. **Evalulating_XLNet_large.ipynb**:  ipynb file of the xlnet large cased model run on google colab notebook. Could not run the code on the cluster as memory specification was exceeding with 16 batch size (cannot change the batch size because we have kept the initial setting of all the experiments same).

15. **Evalulating_XLNet_lemmatization.ipynb**: ipynb file of the xlnet base cased run with lemmatized reviews. Contains both the code and the output.

## Problems faced while performing the experiments



*   **Size of the models**: Most of the models used for the experiments have quite large GPU memory requirements. So while running the experiments on cluster, some of them were facing **OutofMemoryError** in the cluster. As we tried to keep the experimental setup for all the experiments same, changing the batch size was not an option for us.That is why we ran those experiments in google colab environment high GPU RAM set up. (Google Colab Pro+) 
*   **Representation techniques**: Representation techniques like TF-IDF, LDA and Byte Pair Encoding cannot be directly applied to models like XLNet and BERT. If we wish to see their efficacy on these models than we would have to first pre-train the model architectures on those representations and then we can test it on the test dataset. However due  to the timing and resource constraints we were not able perform those. 
