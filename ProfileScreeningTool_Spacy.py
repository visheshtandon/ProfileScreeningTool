# -*- coding: utf-8 -*-

def packages():
    
    print("Importing required Packages...")
        
    # Imports all the required packages
    import spacy, random, io, json, string, datetime, os
    from spacy.util import minibatch,compounding
    from spacy.gold import GoldParse
    from spacy.scorer import Scorer
    from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
    
    global datetime, random, spacy, compounding, minibatch, io, os, json, string, GoldParse, Scorer, classification_report, precision_recall_fscore_support, accuracy_score

def declarations():
        
    global entities,nlp,blacklisted_punctuation, printable, model, output_dir,n_iter, train_test,sub, path, dat    
    printable = set(string.printable)
    
    output_dir=os.path,dir(__file__)
    n_iter=10
    train_test="True"
    
    try:
        model=spacy.load(output_dir)
        print("Model Imported")
    except:
        model=None
    
    path='training.json'
         
def clean_text(text):
    return ''.join([''.join([c for c in text])])

def convert_dataturks_to_spacy(json_file_path):
    
    global entities

    sub=[]
    training_data=[]
    with io.open(json_file_path,'r',encoding='utf-8-sig') as f1:
        
        for line in f1:
            data=json.loads(line) 
        for line in data['annotations_and_examples']:
            sub=[]
            text=clean_text(line['content'])
            entities=[]
            for annotation in line['annotations']:
                point = annotation
                labels=point['tag']
                #sub.append(labels)
                sub.extend([labels])
                if not isinstance(labels, list):
                    labels = [labels]
                    
                for label in labels:
                    entities.append((point['start'], point['end']  , label))

            training_data.append((text, {"entities": entities}))
    
    sub=list(set(sub))
    return training_data
    
    
def train_model(json_file_path):
    
    global nlp, train_data, ner 
    training_data=convert_dataturks_to_spacy(json_file_path)
    
    if model is not None:
        nlp=model
        # load existing spaCy model
        print("Loaded model '%s'" % model)

    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    if("ner" not in nlp.pipe_names):
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    train_data=[]
    
    train_data.extend(training_data)
    
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    
    if model is None:
        optimizer = nlp.begin_training()
        print("hello")
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer=nlp.resume_training()
    
    #global losses
    #losses={}
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        nlp.begin_training()
        
        for iter in range(n_iter):
            random.shuffle(train_data)
            for raw_text,entity_offsets in train_data:
                try:
                    doc=nlp.make_doc(raw_text)
                    gold=GoldParse(doc,entities=entity_offsets['entities'])
                    nlp.update([doc],[gold],drop=0.35,sgd=optimizer)
                except:
                    print("Error in CV: ",raw_text[0:30])
            print("Number of Iterations Completed: (",iter+1,"/",n_iter,")")
           
            
    if(output_dir!=''):
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
    return nlp

def test(output_dir):
    
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    for text, _ in train_data:
        doc = nlp2(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
   
def evaluate2(nlp,examples):
    global d,file
    
    file=open("C:/Users/visheshtandon1/Downloads/Evaluation Scores.txt",mode='a')
    c=0        
    for text,annot in examples:

        doc_to_test=nlp(text)
        
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[0,0,0,0,0,0]

        for ent in doc_to_test.ents:
            
            doc_gold_text= nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
            y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
            
            if(d[ent.label_][0]==0):
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)
                d[ent.label_][0]=1
                d[ent.label_][1]+=p
                d[ent.label_][2]+=r
                d[ent.label_][3]+=f
                d[ent.label_][4]+=a
                d[ent.label_][5]+=1
        c+=1
      
    file.write("Date: "+str(datetime.datetime.today())+"\n\n")

    for i in d:
        file.write("\tEntity: "+i)
        file.write("\n\tAccuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        file.write("\n\tPrecision : "+str(d[i][1]/d[i][5]))
        file.write("\n\tRecall : "+str(d[i][2]/d[i][5]))
        file.write("\n\tF-score : "+str(d[i][3]/d[i][5])+'\n\n')    
    
    file.close()
  
def evaluate1(ner_model, examples):
    scorer = Scorer()

    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
        
    return scorer.scores    

'''
def run_model(model):
    print('Running against the trained model')
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)
    print()
    
    for resume in getResume():
        doc = nlp(resume)
        if(len(doc.ents)>0):
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print()
'''

packages()
declarations()
model=train_model(path)
test(output_dir)
evaluate2(model,train_data)