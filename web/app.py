import os
from flask import Flask
from flask import url_for, redirect, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired,FileAllowed
from datetime import datetime
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from web import DogModel
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
app.config['SECRET_KEY'] = 'BYg352ZiVVafzG1Frwsj'

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(['jpg', 'png','jpeg'], u'Image only!'), FileRequired(u'File was empty!')],label="Select dog or human image")
  
model  = DogModel()
@app.route("/")
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = UploadForm()
    file_path = 'static/images/Airedale_terrier_00163.jpg'
   
    if form.validate_on_submit():        
        myform = form.photo.data
        filename = secure_filename(myform.filename)
        file_path = os.path.join(
            'static', 'images', filename
        )
        
        myform.save(file_path)               
        prediction = model.detect_dog_human(file_path)        
        return render_template('predict.html', form=form,file_path=file_path,prediction=prediction)

    return render_template('predict.html', form=form)

# Uncomment below to run on local machine
if __name__ == "__main__":
  app.run( port=3005, debug=False, host='localhost')


