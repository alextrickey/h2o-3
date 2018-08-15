import sys, os
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from random import randint
import re

'''
PUBDEV-5826: GLRM model predict and mojo predict differ

During training, we are trying to decompose a dataset A = X*Y.  The X and Y matrices are updated iteratively until
the reconstructed dataset A' converges to A using some metrics.

During prediction/scoring, we try to do A = X'*Y where A and Y are known.  H2O will automatically detect if 
A is the same dataset used to train the model.  If it is, it will automatically return X.  If A is not the same
dataset used during training, we will try to get X' given A and Y.  In this case, we will still use the GLRM framework
of update.  However, we only perform update on X' since Y is already known.

At first glance, we will expect that predict on the training frame and predict on a brand new frame will give 
different results since two different procedures are used.  Here, I am trying to narrow the differences between the
two X generated in this case.

I will use a numerical dataset benign.csv and then work with another dataset that contains both numerical and enum
columns prostate_cat.css
'''
def glrm_mojo():
    h2o.remove_all()
    # dataset with numerical values only
    train = h2o.import_file(pyunit_utils.locate("smalldata/logreg/benign.csv"))
    test = h2o.import_file(pyunit_utils.locate("smalldata/logreg/benign.csv"))
    get_glrm_xmatrix(train, test)
    test_glrm_predict(train, test, 2)

    # dataset with enum and numerical columns
    train = h2o.import_file(pyunit_utils.locate("smalldata/prostate/prostate_cat.csv"))
    test = h2o.import_file(pyunit_utils.locate("smalldata/prostate/prostate_cat.csv"))
    test_glrm_predict(train, test, 2)

def get_glrm_xmatrix(train, test):
    x = train.names
    transform_types = ["NONE", "STANDARDIZE", "NORMALIZE", "DEMEAN", "DESCALE"]
    transformN = transform_types[randint(0, len(transform_types)-1)]
    # build a GLRM model with random dataset generated earlier
    glrmModel = H2OGeneralizedLowRankEstimator(k=3, transform=transformN, seed=12345)
    glrmModel.train(x=x, training_frame=train)
    glrmTrainFactor = h2o.get_frame(glrmModel._model_json['output']['representation_name'])

    save_GLRM_mojo(glrmModel) # save mojo model

    MOJONAME = pyunit_utils.getMojoName(glrmModel._id)
    TMPDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "results", MOJONAME))
    h2o.download_csv(test[x], os.path.join(TMPDIR, 'in.csv'))  # save test file, h2o predict/mojo use same file

    frameID, mojoXFactor = pyunit_utils.mojo_predict(glrmModel, TMPDIR, MOJONAME, glrmReconstruct=False) # save mojo XFactor
    glrmTestFactor = h2o.get_frame("GLRMLoading_"+frameID)   # store the x Factor for new test dataset
    print("Comparing mojo x Factor and model x Factor ...")
    pyunit_utils.compare_frames_local(glrmTestFactor, mojoXFactor, 1, tol=1e-10)

def save_GLRM_mojo(model):
    # save model
    regex = re.compile("[+\\-* !@#$%^&()={}\\[\\]|;:'\"<>,.?/]")
    MOJONAME = regex.sub("_", model._id)

    print("Downloading Java prediction model code from H2O")
    TMPDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "results", MOJONAME))
    os.makedirs(TMPDIR)
    model.download_mojo(path=TMPDIR)    # save mojo

def test_glrm_predict(train, test, tolerance):
    x = train.names

    transform_types = ["NONE", "STANDARDIZE", "NORMALIZE", "DEMEAN", "DESCALE"]
    transformN = transform_types[randint(0, len(transform_types)-1)]

    # build a GLRM model with random dataset generated earlier
    glrmModel = H2OGeneralizedLowRankEstimator(k=3, transform=transformN, max_iterations=10, loading_name="xfactors",
                                               seed=12345, init="random")
    glrmModel.train(x=x, training_frame=train)
    pred2 = glrmModel.predict(test) # predict using mojo
    pred1 = glrmModel.predict(train)    # predict using the X from A=X*Y from training

    pyunit_utils.compare_frames_local(pred2, pred1, 1, tol = tolerance) # compare the two reconstructed frames

if __name__ == "__main__":
    pyunit_utils.standalone_test(glrm_mojo)
else:
    glrm_mojo()
