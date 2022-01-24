from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class CallbackHandler():
    def __init__(self,cbs=None):
        self.cbs = cbs if cbs else []

    def cmatrix(pred,true,labels=None):
        if not confusion_matrix: 
            print("is Sklearn conf matrix installed?")
            return False
        #return confusion_matrix(y_true=true,y_pred=pred,labels=labels)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=true,y_pred=pred)
        return disp

    def accuracy(pred,true):
        correct = (pred == true).sum()
        acc = int(correct) / int(len(true))
        print(f'accuracy: {acc:.4f}')

        
