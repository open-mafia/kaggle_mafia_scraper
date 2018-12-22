print("Initializing...")

import os
from datetime import datetime

import numpy as np
import pandas as pd

#---------------------------------------------------------------------
# Helper functions
#---------------------------------------------------------------------
def Timestamp(pr=True):
    ts = str('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.now()))
    if pr: 
        print(ts)
    return ts

#---------------------------------------------------------------------
# Text parsing definitions
#---------------------------------------------------------------------

import xlrd
import csv
from os import sys

def csv_from_excel(excel_file,out_dir):
    workbook = xlrd.open_workbook(excel_file)
    all_worksheets = workbook.sheet_names()
    for worksheet_name in all_worksheets:
        worksheet = workbook.sheet_by_name(worksheet_name)

        if worksheet.nrows == 0: continue

        csv_file = open(out_dir+"\\"+worksheet_name+".csv", 'w', newline="")
        wr = csv.writer(csv_file)#, quoting=csv.QUOTE_ALL)

        for rownum in range(worksheet.nrows):
            row = worksheet.row_values(rownum)
            wr.writerow(row)
        csv_file.close()

def UnpackExcel(in_dir,out_dir):
    """Unpack xls and xlsx files into csv for parsing. """
    for parse_f in os.listdir(in_dir):
        ext=os.path.splitext(parse_f)[1]
        if ext==".xls" or ext==".xlsx":
            print("> Excel File: " + parse_f)
            csv_from_excel(in_dir+"\\"+parse_f,out_dir)
        else:
            print("> File: " + parse_f + " (skipped)")
    pass




#---------------------------------------------------------------------
# Text parsing definitions
#---------------------------------------------------------------------

try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup
import re
import urllib.request

class Post:
    """Class for holding a single post. 
    Attributes:
        user: name of poster.
        raw: entire HTML of the post.
        text: plaintext of the post.
        """
    def __init__(self,bs=None,html=None):
        if bs==None and html==None:
            """Returns empty Post object... why would you do this? """
            self.user=""
            self.text=""
            self.raw=""
            return
        if bs==None:
            """Returns a full Post object, based on unparsed HTML (hope this is right!) """
            self.__init__(self,BeautifulSoup(html,"html5lib"))
            return
        #ELSE:
        """Returns a full Post object, based on BeautifulSoup's parsed HTML. """
        QuoteStr = " AAAQUOTE"

        self.user=bs.find("div",{'class':'poster'}).find("a").text #seems good
        self.wrapped=bs.find("div",{'class':'post'}).find("div",{'class':'inner'})
        self.raw=str(self.wrapped)
        
        quote_cnt=0
        while True:
            try:
                self.wrapped.find("div",{"class":"quoteheader"}).extract() #remove quote headers
            except:
                break
        while True:
            try:
                self.wrapped.find("blockquote").extract() #quote text
                quote_cnt += 1
                #self.wrapped.find("blockquote",{"class":"bbc_standard_quote"}).extract() #quote text
            except:
                break
        while True:
            try:
                self.wrapped.find("div",{"class":"quoteheader"}).extract() #remove quote footers
            except:
                break
        self.text=self.wrapped.getText(" ")
        for i in range(0,quote_cnt):
            self.text += QuoteStr
        #!implement better text handling, because "quotes" and stuff mess this up...
    def __str__(self):
        """Returns string representation """
        return self.user + "\n" + self.text + "\n" + self.raw

class ForumPageReader:
    """Class for reading all the posts from a URL.
    Attributes:
        path: url or file name
        html: raw html
        parsed_html: BeautifulSoup representation of html
        Posts: list of held posts
        Users: set of usernames for posts
    """
    def __init__(self,path,is_url=True):
        """Reads and parses a single forum page. """
        self.path=path
        if is_url:
            html_file = urllib.request.urlopen(path)
        else:
            html_file = open(path)
        self.html = html_file.read()
        self.parsed_html = BeautifulSoup(self.html,"html5lib")

        wrapped_posts = self.parsed_html.find("div",{"id":"forumposts"}).find("form").children
        self.Posts = []
        for wp in wrapped_posts:
            try:
                self.Posts.append(Post(wp))
            except:
                pass
        self.Users = set() #{} is dict
        for p in self.Posts:
            self.Users.add(p.user)

    def __str__(self):
        """Return as a string"""
        return self.path+"\n\n"+str(self.parsed_html)

    def output(self,file): 
        """Output html to file"""
        outf=open(file,mode='w',encoding="UTF-8")
        outf.write(str(self.parsed_html))

class ForumThreadReader:
    """Class for parsing an entire Bay12 forum thread.
    
    Attributes:
        path: url or file name
        Posts: lits of held posts
        Users: set of usernames for posts
        pagecnt: number of pages in post
    """
    def __init__(self,path):
        """Reads an entire thread. """
        #Note: this is hard!
        pageraders = []
        self.Posts = []
        self.Users = set()
        urls = []

        #extract page count
        html_file = urllib.request.urlopen(path)
        html = html_file.read()
        parsed_html = BeautifulSoup(html,"html5lib")
        page_html = parsed_html.find("div",{"id":"postbuttons"}).find("div",{"class":"margintop middletext floatleft"})
        
        nav_pages = page_html.find_all("a",{"class":"navPages"})
        num_pages = 0
        for p in nav_pages:
            x = int(p.text)
            if (x>num_pages):
                num_pages = x
        #print(num_pages) #TEST
        self.pagecnt=num_pages
        
        #base_url = path[:-3] #dirty, works only on base
        base_regex = re.compile(r".*topic=[0-9]+\.") #r".*topic=[0-9]+\."
        base_url=base_regex.findall(path)[0]
        #base_url = _.group(0)

        for i in range(0,self.pagecnt+1):
            urls.append(base_url + str(i*15))
        
        print("Working... (" + str(self.pagecnt)+" pages) @ " + Timestamp(False))
        
        for url in urls:
            pr=ForumPageReader(url,True)
            pageraders.append(pr)
            self.Posts.extend(pr.Posts)
            self.Users = self.Users.union(pr.Users)
            pass
        pass
    pass

class LabeledGame:
    """Encapsulates a labeled game 
    Attributes:
        Users
        Texts
        Labels
        Url
    """

    @staticmethod
    def get_alignments(lines):
        res = {} #{user:label}
        replacements = {} #{original:replacement}
        for line in lines:
            #!IMPLEMENT parse the line, add "user":"label" to line
            pts = line.split(",")
            if pts[1].strip()=="r":
                replacements[pts[0]] = pts[2].strip()
            else:
                res[pts[0]] = pts[1].strip() #!TEMP just user:label for now...
        
        while True:
            failed = {}
            for orig in replacements:
                rep = replacements[orig]
                try:
                    res[orig] = res[rep]
                except:
                    failed[orig] = rep
                    pass
            if failed=={}: break
            replacements = failed.copy()
        return res
    #
    def __init__(self,init_file_path=None,url=None,users=None,texts=None,labels=None,ignoreCase=True,concatenateText=False):
        """Initialize from initialization file (csv) or ready lists. """        
        if init_file_path==None:
            self.Url = url
            self.Users = users
            self.Texts = texts
            self.Labels = labels
            return
        """Init from file """
        tmp_labels = {} #read other lines; {"user":"label"}
        with open(init_file_path) as init_file:
            rawtext = init_file.read()
            if ignoreCase: rawtext = rawtext.lower()
            lines = rawtext.splitlines()
            
            self.Url = lines[0] #read first line
            lines = lines[1:]
            tmp_labels = self.get_alignments(lines)
        #

        ftr=ForumThreadReader(self.Url)

        #
        self.Users = list(ftr.Users)
        self.Labels = []
        for u in self.Users:
            if ignoreCase: u = u.lower()
            try:
                self.Labels.append(tmp_labels[u])
            except:
                self.Labels.append("o") #just say "ooops!"

        self.Texts = []
        SEP = " || " #space as separator between posts

        if concatenateText:
            #concatenate text together
            for user in self.Users:
                u_text = ""
                for post in ftr.Posts:
                    if (post.user==user or (ignoreCase and post.user.lower()==user.lower())):
                        u_text = u_text + SEP + post.text
                self.Texts.append(u_text)
        else:
            users2 = []
            texts2 = []
            labels2 = []
            #keep posts separate
            for i in range(0,len(self.Users)):
                user=self.Users[i]
                label=self.Labels[i]
                for post in ftr.Posts:
                    if (post.user==user or (ignoreCase and post.user.lower()==user.lower())):
                        texts2.append(post.text)
                        users2.append(user)
                        labels2.append(label)
            self.Users=users2
            self.Texts=texts2
            self.Labels=labels2
        return
    #

    def as_DF(self):
        """Return as flattened Pandas DataFrame. """
        raw_urls = []
        for u in self.Labels:
            raw_urls.append(self.Url)
        raw_data = {"urls":raw_urls,"user":self.Users,"label":self.Labels,"text":self.Texts}
        df = pd.DataFrame.from_dict(raw_data)
        df = df[['urls','user','label','text']]
        return df
    #
    def append_DF(self,df):
        """Returns df + self as DataFrame. """
        return pd.concat([df,as_DF()]).reset_index(drop=True)

def WorkThroughDirectory(dir,timestamp2=False):
    """Read and parse all csv files in dir. """
    if timestamp2:
        s_time=datetime.now()
    
    lgames = []
    for parse_f in os.listdir(dir):

        ext=os.path.splitext(parse_f)[1]
        if ext==".csv":
            print("File: " + parse_f + " @ " + Timestamp(False))
            lgames.append(LabeledGame(init_file_path=dir + "/" + parse_f))
            if timestamp2:
                diff=(datetime.now()-s_time).seconds
                print(" - " + str(diff) + " seconds (" + str(diff/60) + " minutes) passed")
        else:
            print("File: " + parse_f + " (skipped)")
    lgames;
    DATA = pd.concat([lg.as_DF() for lg in lgames])
    DATA = DATA.reset_index(drop=True)

    if timestamp2:
        diff=(datetime.now()-s_time).seconds
        print(str(diff) + " seconds (" + str(diff/60) + " minutes) passed")

    return DATA

def GetRelevantData(RAW_DATA):
    q=RAW_DATA.loc[(RAW_DATA["label"].isin(["t","m"]))]
    FULL_DATA = q.copy().reset_index(drop=True)
    FULL_DATA["textlen"] = [len(s) for s in FULL_DATA["text"]]
    FULL_DATA["lbl"]=(FULL_DATA["label"]=="m")
    FULL_DATA = FULL_DATA[["urls","user","label","lbl","textlen","text"]]
    return FULL_DATA

#---------------------------------------------------------------------
# Classifier/pipeline Training definition
#---------------------------------------------------------------------


def sample(labels,cv_pct,split=None):
    N = len(labels)
    indexes = pd.Series(range(0,N))

    idx_TRAIN = pd.Series(range(0,int(N*(1-cv_pct))))
    idx_CV = pd.Series(range(int(N*(1-cv_pct)),N))

    #idx_TRAIN = np.random.choice(indexes, size = int(N*(1-cv_pct)))
    #idx_CV = np.array(list(set(indexes) - set(idx_TRAIN)))

    return [idx_TRAIN, idx_CV]
def balanced_sample(labels,cv_pct):
    N = len(labels)
    indexes = pd.Series(range(0,N))

    idx0 = indexes[labels==False]
    idx1 = indexes[labels==True]
    sz0 = len(idx0)
    sz1 = len(idx1)
    sz = min(sz0,sz1)

    idx0 = np.random.choice(idx0, size = sz)
    idx1 = np.random.choice(idx1, size = sz)

    idx0_TRAIN = np.random.choice(idx0, size = int(sz*(1-cv_pct)))
    idx1_TRAIN = np.random.choice(idx1, size = int(sz*(1-cv_pct)))
    
    idx0_CV = np.array(list(set(idx0) - set(idx0_TRAIN)))
    idx1_CV = np.array(list(set(idx1) - set(idx1_TRAIN)))

    idx_TRAIN = np.concatenate((idx0_TRAIN,idx1_TRAIN))
    idx_CV = np.concatenate((idx0_CV,idx1_CV))

    return [idx_TRAIN, idx_CV]


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD #instead of PCA - it's LSA

def BuildTextPipeline(ngram_rg=(1,2),min_docs=1,max_feats=None,n_PC=20,alg="SVC"):
    lsa = TruncatedSVD(n_components=n_PC)
    if alg=="SVC":
        clf = SVC()
    if alg=="Bayes":
        clf = GaussianNB();
    if alg=="Tree":
        clf = DecisionTreeClassifier()


    tfidf = TfidfVectorizer(analyzer="word",ngram_range=ngram_rg,min_df=min_docs,max_features=max_feats)
    pipe = Pipeline([('tfidf',tfidf),
                     ('lsa',lsa),
                     ('clf',clf)])
    return pipe


#---------------------------------------------------------------------
# Metrics definition
#---------------------------------------------------------------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import warnings

def OutputMetrics(actual, predicted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Do stuff here
        accuracy = accuracy_score(actual,predicted)
        auc = roc_auc_score(actual,predicted)
        f1s = f1_score(actual,predicted)
        print("Accuracy: " + str(round(accuracy,4)) + " AUC: " + str(round(auc,4)) + " F1: " + str(round(f1s,4)))
    return [accuracy,auc,f1s]

#---------------------------------------------------------------------
# Work Phase
#---------------------------------------------------------------------

#

def _sub_UnpackExcel(input_dir):
    #print("\nUnpacking Excel (.xls, .xlsx) into CSV (.csv).")
    loc_dirs = os.listdir(input_dir)
    print("Directories:\n " + str(loc_dirs))
    xl_dir = input("Input (Excel) directory: ")
    csv_dir = input("Output (CSV) directory: ")
                
    #try to parse index-based
    try: xl_dir = loc_dirs[int(xl_dir)]
    except: pass
    try: csv_dir = loc_dirs[int(csv_dir)]
    except: pass

    #attempt unpacking
    try: UnpackExcel(input_dir + "\\" + xl_dir, input_dir + "\\" + csv_dir) 
    except:
        print("ERROR unpacking.\n")
        return
#

def _sub_ParseCSV(input_dir,data_dir):
    print("\nDownloading and parsing threads from CSV.")
    in_dirs=os.listdir(input_dir)
    print("Input Directories:\n " + str(in_dirs))
    csv_dir = input("Input (CSV) directory: ")
    try: csv_dir = in_dirs[int(csv_dir)]
    except: pass
                
    base_name = input("Data file base name: ")
                
    try:
        RAW_DATA = WorkThroughDirectory(input_dir + "\\" + csv_dir)
        FULL_DATA = GetRelevantData(RAW_DATA)
        DATA = FULL_DATA[["lbl","text"]].copy()
    except:
        print("ERROR downloading/parsing.\n")
        return

    #SAVE:
    try:
        RawFile = data_dir+"\\"+base_name+".raw.csv"
        ShortFile = data_dir+"\\"+base_name+".csv"
        RD=RAW_DATA
        RD["text"]=(RAW_DATA["text"].str.replace(","," "))
        RD.to_csv(RawFile,encoding='utf-8')

        D=DATA
        D["text"]=(DATA["text"].str.replace(","," "))
        D.to_csv(ShortFile,encoding='utf-8')

        print("Saved as " + ShortFile + " and " + RawFile)
    except:
        print("ERROR saving.\n")
        return
#

def _sub_LearnModel(DATA):
    #DATA['lbl'] = DATA['lbl'].values.astype('U')  ## Even astype(str) would work
    DATA.dropna(how="any",inplace=True)

    #SAMPLE
    cv_pct=0.2
    idx = sample(DATA["lbl"],cv_pct)
    #idx = balanced_sample(DATA["lbl"],cv_pct)
    TRAIN = DATA.loc[idx[0]]
    CV = DATA.loc[idx[1]]


    #PARAMS
    Qngram = [(1,1)]
    Qmax_feats=[100,300,1000]
    Qn_PC=[10,30,100,300]
    Qalgs=["SVC","Bayes","Tree"]

    learned_models = {}
    print("[ngrams,features,PC] _alg (metrics):\n")
    for qmax_feats in Qmax_feats:
        for qn_PC in Qn_PC:
            if qn_PC <= qmax_feats:
                for qngram in Qngram:
                    for qalg in Qalgs:
                        print(str([qngram,qmax_feats,qn_PC])+" _"+qalg+"  ",end="")
                        
                        try:
                            #TRAIN
                            pipe = BuildTextPipeline(ngram_rg=(1,1),max_feats=200,n_PC=10,alg=qalg)
                            model = pipe.fit(TRAIN.text,TRAIN.lbl)

                            #EVALUATE
                            CV_predict = model.predict(CV.text)
                            metrics = OutputMetrics(CV.lbl,CV_predict)
                            print("")
                            learned_models[model]=metrics
                        except Exception as e:
                            print("Failed (Error: ", e, " )")
    #
    
    
    return(learned_models)
#

def _sub_LearnModelWrapper(data_dir):
    print("\n!IMPLEMENT load from disk")

    model_prompt="""Models:
    1. By-post"""
    print(model_prompt)


    files=os.listdir(data_dir)
    print("Input Files:\n " + str(files))
    data_file = input("Data file: ")
    try: data_file = files[int(data_file)]
    except: pass
    
    try:
        DATA = pd.read_csv(data_dir + "\\" + data_file, encoding = 'utf8')
        DATA.apply(lambda x: pd.lib.infer_dtype(x.values))
        DATA.dropna(subset=["text"], inplace=True)
    except Exception as e:
        print(e)
        return(None)
    model = _sub_LearnModel(DATA)
    return(model)
#

def DoAll():
    """!DEPRECATED. """
    #SETUP
    CSVDirectory="Input\\short"
    #ExcelDirectory="ToParse2"

    #UNPACK .xls and .xlsx
    #UnpackExcel(ExcelDirectory,CSVDirectory)

    RawOutputSaveFile="LRawOut.csv"
    OutputSaveFile="LOut.csv"


    #DOWNLOAD & PARSE
    RAW_DATA = WorkThroughDirectory(CSVDirectory)
    FULL_DATA = GetRelevantData(RAW_DATA)
    DATA = FULL_DATA[["lbl","text"]].copy()

    #SAVE:
    RAW_DATA["text"].str.replace(","," ").to_csv(RawOutputSaveFile)
    DATA["text"].str.replace(","," ").to_csv(OutputSaveFile)



    #SAMPLE
    cv_pct=0.2
    idx = sample(DATA["lbl"],cv_pct)
    #idx = balanced_sample(DATA["lbl"],cv_pct)
    TRAIN = DATA.loc[idx[0]]
    CV = DATA.loc[idx[1]]

    #TRAIN
    pipe = BuildTextPipeline(ngram_rg=(1,2))
    model = pipe.fit(TRAIN.text,TRAIN.lbl)

    #EVALUATE
    CV_predict = model.predict(CV.text)
    CV_predict
    OutputMetrics(CV.lbl,CV_predict)
#

def menu() :
    menu_prompt = """-------------------------------------------------------------------------
1. Unpack Excel files into CSV
2. Download + parse threads (from CSV) and save data to disk
3. Train a model from a file [!unfinished]
4. <blank>

9. Clear data
0. Exit

> """

    exit_prompt = "Bye!"
    
    input_dir = os.getcwd()+"\\Input"
    data_dir=os.getcwd()+"\\Data"

    models = []

    while True:
        try:
            response = input(menu_prompt)
            if response=="0":
                print(exit_prompt)
                break;
            elif response=="1":
                _sub_UnpackExcel(input_dir)
            elif response=="2":
                _sub_ParseCSV(input_dir,data_dir)
            elif response=="3":
                models.extend(_sub_LearnModelWrapper(data_dir))
                
            elif response=="4":
                print("\n<blank>")

            elif response=="9":
                print("\nClearing models...\n")
                models = []
            else:
                print("\nIllegal input.\n")
            pass
        except Exception as e:
            print("EXCEPTION: ", e)
    

#


menu()
#DoAll()