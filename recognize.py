import cv2
import os
from sklearn.metrics import average_precision_score

def metrics (y_test, y_pred) :
    result= {}
    people = set(y_test)
    
    for person in people:
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0        

        for i in range(len(y_test)):
            if (y_test[i] == y_pred[i] == person):
                TP = TP + 1
            elif (y_test[i] != y_pred[i] and y_test[i] == person):
                FN = FN + 1
            elif (y_test[i] != y_pred[i] and y_pred[i] == person):
                FP = FP + 1
            elif ( (y_test[i] != y_pred[i]) and (y_test[i] != person) and (y_pred[i] != person) ):
                TN =TN + 1
                
        result[person] = {"TP":TP, "TN":TN, "FP":FP, "FN":FN}
        
    return result

def precision (data) :
    precisions = {}
    people = data.keys()
    
    for person in people :
        metrics = data[person]
        precision = (metrics["TP"])/( (metrics["TP"]) + (metrics["FP"]) )
        precisions[person] = precision
    return precisions
            
            

def test(img, classifier, scaleFactor, minNeighbors,clf):
    ids = []
    id = None
    features = classifier.detectMultiScale(img, scaleFactor, minNeighbors)
    for (x, y, w, h) in features:
        #cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        # Predicting the id of the user
        id, _ = clf.predict(img[y:y+h, x:x+w])
        # Check for id of user and label the rectangle accordingly
        ids.append(id)
        break
    return id

    


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        # Predicting the id of the user
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        # Check for id of user and label the rectangle accordingly
        if id==1:
            cv2.putText(img, "Zubayr", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
    
    return coords

# Method to recognize the person
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 0)}
    coords = draw_boundary(img, faceCascade, 1.1, 25, color["white"], "Face", clf)
    return img


# Loading classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier_new2.xml")


#------------------------------------------------
use_camera = 0
use_pictures = 1
#------------------------------------------------

if (use_pictures == 1):
    
    imagePaths= [os.path.join("dataset", f) for f in os.listdir("dataset")]
    results = []
    #Exrtract images from a dataset 
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        
        name = os.path.split(imagePath)[1] 
        image = cv2.imread(imagePath,0)
        result = test(image, faceCascade, 1.1, 25, clf )
        results.append(result)
        #print(result)
    
    y_test = [0]*len(imagePaths)
    
    things = metrics(y_test, results)
    prec = precision(things)
    print("The metrics are :" + str(things))
    print("The precision is :" + str(prec))


if (use_camera == 0):
    # Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
    video_capture = cv2.VideoCapture(0)
    while True:
        # Reading image from video stream
        _, img = video_capture.read()
        # Call method we defined above
        img = recognize(img, clf, faceCascade)
        # Writing processed image in a new window
        cv2.imwrite("test.jpg", img)
        cv2.imshow("face detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if (use_camera == 0):
    # releasing web-cam
    video_capture.release()
    # Destroying output window
    cv2.destroyAllWindows()