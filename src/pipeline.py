#!/opt/miniconda3/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path
import os,sys, glob
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0])
pd.set_option('display.max_columns', None)
################## FUNCTIONS ######################
def style_plot_axes(ax, plot_font, ax_width, grid=False):
    for spine in ax.spines.values():
        spine.set_linewidth(ax_width)

    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        pad=7, length=6, width=1.5, labelsize=plot_font,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True,
        length=3, width=1.2,
    )
    ax.minorticks_on()
    if grid:
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.8)

    return plot_font

def fix_log(axis, ticks, start, end):
    def update_ticks(z, pos):
        if z ==0:
            return '0'
        elif z >=1 and z <1000:
            return '%d' %(z)
        elif z < 1 and z > 0.001:
            return z
        else:
            return  '10$^{%1.0f}$' %(np.log10(z))

    upper = max(float(start), float(end), 0.0)
    minor = []

    if upper >= 1:
        max_exponent = int(np.floor(np.log10(upper)))
        for exponent in range(max_exponent + 1):
            base = 10**exponent
            for multiplier in range(2, 10):
                tick = multiplier * base
                if tick <= upper:
                    minor.append(tick)

    if ticks == "xticks":
        axis.set_xticks(minor, minor=True)
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
    else:
        axis.set_yticks(minor, minor=True)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
#########################################################################



### -- Import data - hard code just now -- ###
notebook_dir = os.path.abspath('')
root_dir = os.path.dirname(notebook_dir)  # Moves up 1 level from 'notebooks' to project root
#print(notebook_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

data_dir = Path(root_dir) / "data"
full_paths = glob.glob(str(data_dir / "*.csv"))
file_mapping = {os.path.basename(f): f for f in full_paths}
csv_files = list(file_mapping.keys())

# print("-"*30,"Available Data Files","-"*30); print(csv_files)
# infile = str(input("Input data file: "))
# check_file = os.path.isfile(infile)
# while check_file == False:
#     print(csv_files)
#     infile = str(input('Input data file: '))
#     check_file = os.path.isfile(infile)
infile = 'synthetic_gambling_aml.csv' # hard code in just now

df = pd.read_csv(data_dir/infile); print(df)

### -- Check for missing values --#
def missing(data):
    cols = data.columns 
    for (i,col) in enumerate(cols):
        print("%s (%s)  - missing values = " %(col, type(data[col].iloc[0])), 
              data[col].isnull().sum())
#missing(df) # ALL GOOD

## --Check the numbers --##
target = 'is_suspicious_activity'

non = df[df[target] == 0]
sus = df[df[target] == 1]
print("-"*80)
print('Total length of dataset = %d, of which %d are suspect and %d are not' 
      %(len(df), len(sus),len(non)))
print("-"*80)

## -- EDA: Histogram function so each feature can be examined in turn --##
def hist(para,dbs,con): # dataframe, parameter of interest and desired bin width
    data = df[para]
    #print(df2345[para].describe())
    
    min_val = np.min(data); max_val = np.max(data)
    dbw = (max_val - min_val)/dbs
    min_boundary = -1.0 * (min_val % dbw - min_val)
    max_boundary = max_val - max_val % dbw + dbw
    n_bins = int((max_boundary - min_boundary) / dbw) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
  
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    font = style_plot_axes(ax, 13, 2)
   
    ## Histograms
    m1= np.mean(non[para]); s1 = np.std(non[para],ddof=1)
    label = "Non-suspect: $\mu = %1.2f, \sigma = %1.2f$" %(m1,s1)
    ax.hist(non[para], bins=bins, color="w", edgecolor='grey', 
            lw=2, label = label)     
                         
    m2 = np.mean(sus[para]); s2= np.std(sus[para],ddof=1)
    label = "Suspect:         $\mu = %1.2f, \sigma = %1.2f$" %(m2,s2)
    ax.hist(sus[para], bins=bins, color="r", edgecolor='k', 
            lw=2, label = label);
    text = '%s' %(para)                   
    plt.ylabel('Number', size=font); plt.xlabel(text, size=font)    
     
    # Need a log y-scale due to differences in sample sizes 
    ax.set_yscale('symlog');

    ax.set_xlabel(para, size=font)
    ax.set_ylabel("Number", size=font)
    y1, y2 = ax.get_ylim()
    fix_log(ax, "yticks", y1, y2)

    ax.legend(fontsize = 0.7*font,loc="upper right")
    plt.tight_layout()
    plt.show(); plt.close()
    
    ### A/B testing between the two classes
    from scipy.stats import norm
    #con = 95
    p = 0.5*(1-con/100) # 2-sided
    alpha = 0.5-p
    m = np.abs(m1 - m2)
    n1 = len(non); n2 = len(sus)
    
    pooled_var = s1**2/n1 + s2**2/n2
    pooled_SD = pooled_var**0.5
    pooled_SE = pooled_SD*(1/n1 + 1/n2)**0.5
    Z = norm.ppf(1-p,loc=0,scale=1)
    CI = Z*pooled_SD
    
    print("\n----------- For %s -----------" %(para))
    
    print("\nAt %1.1f%% confidence the difference in the sample means is %1.2f +/- %1.2f" %(con,m,CI))
    if (m-CI < 0) & (m+CI >0):
        sig = 'passes through 0, so not statistically significant'
    else:
        sig = "doesn't passes through 0, so statistically significant"
    print("(%1.2f to %1.2f) - %s" %(m-CI,m+CI, sig))  

#hist('customer_age',10,95) 

print("--"*10, "Machine learning", "--"*10)

cols = df.columns; # print(cols)
test_frac = 0.2

X = df.drop(columns=target)
y = df[target]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                stratify=y, # preserves the fraud ratio
                                                random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

N = [x for x in y_train if x==0]; S = [x for x in y_train if x==1]
print('Training data have %d suspect and %d non-suspect players' %(len(N),len(S)))
N = [x for x in y_test if x==0]; S = [x for x in y_test if x==1]
print('Test data have %d suspect and %d non-suspect players' %(len(N),len(S)))


from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from sklearn.metrics import precision_recall_curve,ConfusionMatrixDisplay,roc_curve,auc, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth = 5),
    "Logistic Regression": LogisticRegression(C=0.08858667, solver='newton-cg'),
    "Support Vector": SVC(C=1, gamma=0.1,probability=True), 
    'XGBoost': XGBClassifier(n_estimators=500,max_depth=5,learning_rate=0.03,
                             subsample=0.8,colsample_bytree=0.8,random_state=42)
    }


def ML(cl,desc,threshold,confidence,feature_importance,tuning,show_plot):   
    #threshold = 0.5 # th1 + float(th2)/100
    classifier = classifiers.get(cl) 
    if desc != "Class Weighting":
        classifier.set_params(class_weight=None)
        if desc == 'SMOTE':
            sampler = SMOTE()
        else:
            sampler = RandomUnderSampler(sampling_strategy='majority') 
        model = Pipeline([(desc,sampler),(cl, classifier)])
    else:
        if cl != 'XGBoost':
            classifier.set_params(class_weight='balanced')
        else:
            ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
            classifier.set_params(scale_pos_weight=ratio) 
            #print('****blah****', classifier)
        desc = "class"
        model = classifier
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:,1] 
    #the model’s estimated probability that a transaction is fraud
    if tuning != 'n':
        f1_scores = []
        all_scores = []
        thresholds = np.linspace(0.01,0.99,99) # FINE TUNING
        for t in thresholds:
            y_pred = (y_prob > t).astype(int)
            f1_scores.append(f1_score(y_test, y_pred))
            all_scores.append(t)
            all_scores.append(f1_score(y_test, y_pred))
            all_scores.append(precision_score(y_test, y_pred))
            all_scores.append(recall_score(y_test, y_pred))
            alerts = y_pred.sum()
            all_scores.append(alerts/len(y_pred))
            
            #y_pred = (y_prob > threshold).astype(int)
        best = thresholds[np.argmax(f1_scores)]
        all_scores = np.reshape(all_scores,(-1,5));
        th_df = pd.DataFrame(all_scores, columns=['threshold','f-score','precision','recall','alert']);
        th_df.to_csv('thresholds.csv')
        print('================= TRESHOLD TUNING ===================================')
        print("For %s with %s imbalance correction, highest F1 (= %1.3f) at threshold = %1.3f" 
              %(cl,desc,np.max(f1_scores),best))
        print('Data written to thresholds.csv')
        print("======================================================================")
    
    y_pred = (y_prob > threshold).astype(int)
    #the final classification decision after applying a threshold
    y_score = classifier.predict_proba(X_test)[:, 1]
    #return cl,desc,threshold,y_test,y_pred,y_prob,y_score
    
    ## PUTTING METRIC CODE HERE - MAKES THIS CELL VERY LONG, BUT LOSE INTERACTIVITY OTHERWISE
    from sklearn.metrics import precision_recall_curve,ConfusionMatrixDisplay,roc_curve, auc
    #Evaluate Performance
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("For %s resampled via %s with classification threshold = %1.2f" %(cl,desc, threshold))
    print("---------------------------------------------------------------------")
    print("Precision = %1.3f" %(precision)); print("Recall  = %1.3f" %(recall))
    print("F1 score = %1.3f" %(f1))
    total = tp + fp + tn + fn
    print('\nNo. true positives (TP)  = %d (%1.2f%%)' %(tp,100*tp/total))
    print('No. false positives (FP)  = %d (%1.2f%%)' %(fp,100*fp/total))
    print('No. true negatives (TN)  = %d (%1.2f%%)' %(tn,100*tn/total))
    print('No. false negatives (FN) = %d (%1.2f%%)' %(fn,100*fn/total))

    tr = 0.5
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr) # TPR (True Positive Rate) → y-axis  FPR (False Positive Rate) → x-axis
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
    
    font = 12
    plt.rcParams.update({'font.size': font})

    figsize = 12,10; gridsize = 20,20
    fig = plt.figure(figsize=(figsize))
    
    ax1 = plt.subplot2grid((gridsize), (0, 0), colspan=8, rowspan = 8)
    ax2 = plt.subplot2grid((gridsize), (0, 9), colspan=8, rowspan = 6)
    ax3 = plt.subplot2grid((gridsize), (8, 9), colspan=8, rowspan = 6)
    ax4 = plt.subplot2grid((gridsize), (10 ,1), colspan=6, rowspan = 4)
    
    disp = ConfusionMatrixDisplay.from_predictions(y_test,y_pred,
                                                   normalize="true",
                                                   ax=ax1,cmap=plt.cm.Blues,
                                                   display_labels=['Non-suspect', 'Suspect'], 
                                                   colorbar=False)

    ax1.tick_params(axis='both', which='major', labelsize=font,pad=10) 
    plt.setp(ax1.get_yticklabels(), rotation=90, ha='center')

    for labels in disp.text_.ravel(): # FOR NUMBERS IN BOXES
        labels.set_fontsize(font)
        labels.set_text(f"{float(labels.get_text()):.4f}")

    ax1.set_xlabel('Predicted outcome',fontsize=font, labelpad=10)
    ax1.set_ylabel('Actual outcome',fontsize=font, labelpad=10)
    ax1.set_title("%s validation" %(cl), fontsize=font)

    # PRECISION-RECALL CURVE
    ax2.plot(recall_vals, precision_vals, lw =3, color = 'r', 
             label = '%s precision-recall curve' %(cl))
    ax2.plot([0, 1], [1, 1], linestyle='--', color='k', lw=2,
             label='Ideal Classifier (Precision=1.0)')
    plt.setp(ax2.spines.values(),linewidth=2)
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.legend(fontsize = 0.8*font,loc='upper right',labelcolor='k')

    ## ROC CURVE
    

    plt.setp(ax3.spines.values(),linewidth=2)
    ax3.plot(fpr, tpr, linestyle = '-',  lw =3, color = 'r', zorder =2, label="ROC: AUC = %1.2f" %(roc_auc))
    ax3.plot(fpr, fpr, linestyle = '--',  lw =2, color = 'k', zorder = 2,label="%1.1f benchmark" %(tr))
    ax3.fill_between(fpr, y1 = 0, y2 = tpr, where= (fpr > 0) & (fpr < tpr),color= "lightgray",zorder=1)
    ax3.set_xlabel('False positive rate'); ax3.set_ylabel('True positive rate')
    ax3.legend(fontsize = 0.8*font,loc='upper right')

    pr_auc = average_precision_score(y_test, y_prob)
    base_rate = y_test.mean()
    precision_lift = 0.200 / base_rate
    
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Base rate: {base_rate:.4f}")
    print(f"Precision lift over base rate: {precision_lift:.1f}x")

    ## Diagnostic Plot - ERROR BARS BETTER
    plt.setp(ax4.spines.values(),linewidth=2)
    sus_probs = y_prob[y_test == 1]
    non_probs = y_prob[y_test == 0]

    from scipy.stats import norm
    #confidence = 99
    p = 1-confidence/100
    alpha = 0.5-p/2

    def stats(data):
        m = np.mean(data)
        n = len(data)
        s = np.std(data,ddof=1)
        Z = norm.ppf(1-p,loc=0,scale=1)
        SE = s/(float(n)**0.5)
        CI = Z*SE

        return m,n,Z,s,CI

    y_numeric = [0.3, 0.7]
    y_words = ['Suspect', 'Non-suspect']
    
    m,n,Z,s,CI = stats(sus_probs)
    print("\nSus: %1.2f%% confidence,the mean is %1.2f +/- %1.2f (range of %1.2f to %1.2f)"
              %(confidence,m,CI,m-CI,m+CI))

    ax4.errorbar(m,y_numeric[0], xerr=CI, fmt='o', color='r', capsize=5, label= "$%1.3f\pm%1.3g$" %(m,CI))

    m,n,Z,s,CI = stats(non_probs)
    print("Non: %1.2f%% confidence, the mean is %1.2g +/- %1.2g (range of %1.2g to %1.2g)"
              %(confidence,Z,CI,m-CI,m+CI))

    ax4.errorbar(m, y_numeric[1], xerr=CI, fmt='o', color='g', capsize=5, label= "$%1.3f\pm%1.3g$" %(m,CI))

    ax4.set_ylim(0,1)
    ax4.set_xlabel("Predicted laundering probability (%1.2f%% confidence)" %(confidence));
    plt.yticks(y_numeric, y_words)
    ax4.legend(fontsize = 0.8*font,loc='upper right')
    #plt.tight_layout()

    if show_plot != 'n':
        print('The confusion matrix shows the normalised values')
        plt.show()
    
    ###### FEATURE IMPORTANCE ####
    if feature_importance != 'n': 
        print("\nMay take a few minutes to show feature importance")
        from sklearn.inspection import permutation_importance
        results = permutation_importance(model, X_train, y_train, scoring='accuracy')
        importance = results.importances_mean
        features = df.drop(target,axis = 1) 
        features = features.columns.tolist()
        df1 = pd.DataFrame(features, columns=['Feature']); 
        df2 = pd.DataFrame(importance, columns=['Importance'])
        df1['Importance'] = df2.Importance  # adding to df1
        #print(df1.sort_values(by=['Importance'], ascending=False)) 
        if feat == 'Alphabetically':
            tmp = df1.sort_values(by=['Feature'], ascending=False)
        else:    
            tmp = df1.sort_values(by=['Importance'], ascending=True)

        #font = 12
        #plt.rcParams.update({'font.size': font})
        plt.figure(figsize=(6,5))
        ax = plt.gca()
        plt.barh(tmp['Feature'],tmp['Importance'],fc="silver", ec='b', label=dd_samp.value)
        plt.title('%s Feature Importance' %(dd_ML.value),fontsize=font)
        plt.legend(fontsize = 0.8*font,loc='upper right')    
        plt.show()

ML('XGBoost','Class Weighting',threshold = 0.65,confidence = 95,feature_importance = 'n',tuning = 'n',show_plot = 'y') # threshold = 0.65 FROM tuning = 'y'
