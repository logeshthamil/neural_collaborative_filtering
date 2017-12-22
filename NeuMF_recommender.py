
import os
import random
import time

import pymongo.errors as pyerror
import pymongo
import pandas
import numpy
from scipy.sparse import csr_matrix
import implicit
from keras import initializations
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop


class RecommendationEngine(object):
    """
    This is the base class for the Recommendation Engine. The implementation of this abstract class
    should be overridden by the derived class
    """

    def __init__(self,
                 num_recommendations=5,
                 allproducts_collectionname="product",
                 trainingset_collectionname="trainingdata",
                 event_collectionname="inputs",
                 recommendation_collectionname="outputs",
                 dataformat="None",
                 opdatadir="/tmp/testrecommender/ldarec/"):
        """
        :param num_recommendations: the number of recommendations
        :param allproducts_collectionname: the allproducts collectionname
        :param trainingset_collectionname: the allevents collectionname
        :param event_collectionname: the input collectionname
        :param recommendation_collectionname: the output collectionname
        :param dataformat (str): the data format used by the recommender
        :param opdatadir: opdatadir to store the intermediate files
        """
        self._num_recommendations = num_recommendations
        self._allproducts_collectionname = allproducts_collectionname
        self._trainingset_collectionname = trainingset_collectionname
        self._event_collectionname = event_collectionname
        self._recommendation_collectionname = recommendation_collectionname
        self._mongousername = ""
        self._mongopassword = ""
        self._mongoauthentication = ""
        self._mongoip = ""
        self._opdatadir = opdatadir
        self._outputlogfile = self._opdatadir
        self._recenginerunning = False
        self.visualizemodel = False
        self._uservisits = pandas.DataFrame()
        self._allitems = pandas.DataFrame()
        self.dataformat = dataformat
        self.data_manager = None
        # make Rec output dir if not found
        if not os.path.exists(self._opdatadir):
            os.makedirs(self._opdatadir)
        self._mongouri = None
        self._dbname = None

    @staticmethod
    def recommendation_engine_factory(recommender_name, parameter_dict=None):
        """
        A function factory which returns the object of the recommendation engine.
        :param recommender_name: the recommendation engine class name
        :param parameter_dict: the dictionary of parameters in the format {parameter_name: parameter_value}
        :return: the object of the recommendation engine
        """
        recommender = eval(recommender_name)(**parameter_dict)
        if not isinstance(recommender, RecommendationEngine):
            print("RecommendationEngine - the given recommender ({}) is not a valid".format(recommender_name))
        else:
            return recommender

    def setuservisits_and_allitems(self, uservisits=None, allitems=None):
        """
        Set the training set of the RecommendationEngine
        :param uservisits: the user visits dataframe
        :param allitems: the all items dataframe
        """
        if uservisits is not None:
            self._uservisits = uservisits
        if allitems is not None:
            self._allitems = allitems

    def setdb_params(self, mongouri=None, dbname="testrecommender"):
        """
        A method to set the db parameters
        :param mongouri: the mongo uri
        :param dbname: the database name
        """
        if mongouri is not None:
            self._mongouri = mongouri
        if dbname is not None:
            self._dbname = dbname
        self._outputlogfile = self._outputlogfile + dbname + "_outputlog.txt"

    def _create_cappedcollection(self):
        """
        A method to create a capped collection to fetch all the input events
        """
        client = pymongo.MongoClient(self._mongouri)
        inputdb = client[self._dbname]
        try:
            inputdb.create_collection(name=self._event_collectionname, capped=True, size=10000, max=100)
            inputdb[self._event_collectionname].insert_one({"userid": "temp", "itemid": "temp"})
        except pymongo.errors.CollectionInvalid:
            pass

    def update_model(self):
        raise NotImplementedError("This method should be overridden by the derived class")

    def start_recommender(self, numRecs=None):
        raise NotImplementedError("This method should be overridden by the derived class")

    def stop_recommender(self):
        raise NotImplementedError("This method should be overridden by the derived class")

    def insertandretrieve_recommendation_fromcc(self, userid=None, itemid=None):
        """
        A method to insert the recommendation and get the recommendations from capped collecton
        :param userid: the userid
        :param itemid: the itemid
        """
        client = pymongo.MongoClient(self._mongouri)
        outputcollection = client[self._dbname][self._recommendation_collectionname]
        outputcollection.drop()
        if self._recenginerunning is True:
            inputcollection = client[self._dbname][self._event_collectionname]
            inputcollection.insert_one({"userid": str(userid), "itemid": str(itemid)})
            rec_calculated = False
            recommendations = []
            while not rec_calculated:
                try:
                    recommendations = outputcollection.find_one({"userid": str(userid)}).get("recs")
                    rec_calculated = True
                except pyerror.CursorNotFound:
                    pass
            return recommendations
        else:
            raise BrokenPipeError("The recommendation engine is not running")

    def insertandretrieve_recommendation_fromlog(self, userid=None, itemid=None):
        """
        A method to insert the recommendation and get the recommendations from the log file
        :param userid: the userid
        :param itemid: the itemid
        """
        logfile = open(self._outputlogfile, 'w')
        logfile.close()
        if self._recenginerunning is True:
            client = pymongo.MongoClient(self._mongouri)
            inputcollection = client[self._dbname][self._event_collectionname]
            inputcollection.insert_one({"userid": str(userid), "itemid": str(itemid)})
            rec_calculated = False
            recommendations = []
            while not rec_calculated:
                logfile = open(self._outputlogfile, 'r')
                loglist = reversed(logfile.readlines())
                logfile.close()
                tofind = "userid: " + str(userid)
                reccs = []
                rec_calculated = False
                for line in loglist:
                    if (tofind in line) and ("recommendations:" in line):
                        rec_calculated = True
                        recs = line.split(" recommendations: ")[1].split(", recommendation calculation time")[0]
                        reccs = recs.split(",")
                        break
                if rec_calculated:
                    for r in reccs:
                        recommendations.append(r)
            return recommendations
        else:
            raise BrokenPipeError("The recommendation engine is not running")


class ImplicitLSRecommendationEngine(RecommendationEngine):
    """
    An implementation of the Implicit feedback Least Square recommendation engine.
    """

    def __init__(self, **kargs):
        """
        :param num_recommendations: the number of recommendations
        :param num_topics: the number of topics
        """
        num_recommendations = 5
        numtopics = 20
        if kargs.get('num_recommendations') is not None:
            num_recommendations = kargs.get('num_recommendations')
        if kargs.get('numtopics') is not None:
            numtopics = kargs.get('numtopics')
        self.dataformat = "bag_of_word_in_memory"
        super().__init__(num_recommendations=num_recommendations, dataformat=self.dataformat)
        self.__itemuser = None
        self.__useritem = None
        self.__recommender = None
        self.__useridx2id = {}
        self.__userid2idx = {}
        self.__itemidx2id = {}
        self.__itemid2idx = {}
        self.__numtopics = numtopics

    def update_model(self):
        """
        A method to update the model of the recommender
        """
        for itemidx, itemid in self._allitems.iterrows():
            self.__itemid2idx[str(itemid['itemid'])] = itemidx
            self.__itemidx2id[itemidx] = str(itemid['itemid'])
        for useridx, userid in enumerate(self._uservisits['userid'].unique()):
            self.__userid2idx[str(userid)] = useridx
            self.__useridx2id[useridx] = str(userid)
        userid = self._uservisits['userid'].values
        itemid = self._uservisits['itemid'].values
        rating = self._uservisits['rating'].values
        useridx = [self.__userid2idx[str(int(uid))] for uid in userid]
        itemidx = [self.__itemid2idx[str(int(iid))] for iid in itemid]
        rating = list(map(numpy.double, rating))
        self.__itemuser = csr_matrix((rating, (useridx, itemidx)), shape=(len(set(useridx)), len(set(itemidx))))
        self.__recommender = implicit.als.AlternatingLeastSquares(factors=self.__numtopics)
        self.__recommender.fit(self.__itemuser)

    def start_recommender(self, numRecs=None):
        """
        A method to start the recommender
        :param numRecs: number of recommendations
        """
        if numRecs is not None:
            self._num_recommendations = numRecs
        self.__useritem = self.__itemuser.T.tocsr()

    def __insertandretrieve_recommendation(self, userid=None, itemid=None):
        """
        A method to insert and retrieve recommendation.
        :param userid: the userid
        :param itemid: the itemid
        :return: the recommendations
        """
        def recommendation2rec(recommendationsip=None):
            recs = []
            for recommendation in recommendationsip:
                recs.append(self.__itemidx2id[recommendation[0]])
            return recs
        userid = str(userid)
        itemid = str(itemid)
        if userid in list(self.__userid2idx.keys()):
            useridx = self.__userid2idx[userid]
            recommendations = self.__recommender.recommend(useridx, self.__useritem, N=self._num_recommendations)
            recommendations = recommendation2rec(recommendationsip=recommendations)
        else:
            if itemid in list(self.__itemid2idx.keys()):
                itemidx = self.__itemid2idx[itemid]
                recommendations = self.__recommender.similar_items(itemidx, N=self._num_recommendations)
                recommendations = recommendation2rec(recommendationsip=recommendations)
            else:
                recommendations = list(self.__itemid2idx.keys())
                random.shuffle(recommendations)
                recommendations = recommendations[:self._num_recommendations]
        return recommendations

    def insertandretrieve_recommendation_fromlog(self, userid=None, itemid=None):
        return self.__insertandretrieve_recommendation(userid=userid, itemid=itemid)

    def insertandretrieve_recommendation_fromcc(self, userid=None, itemid=None):
        return self.__insertandretrieve_recommendation(userid=userid, itemid=itemid)

    def stop_recommender(self):
        self.__useritem = csr_matrix((0, 0))


class NeuMFRecommendationEngine(RecommendationEngine):
    """
    An implementation class of the Neural Matrix Factorization recommendation engine.
    """

    def __init__(self, **kargs):
        """
        :param num_recommendations: the number of recommendations
        :param num_topics: the number of topics
        """
        num_recommendations = 5
        numtopics = 20
        numiterations = 100
        batchsize = 1000
        learningrate = 0.001
        learner = "adam"
        if kargs.get('num_recommendations') is not None:
            num_recommendations = kargs.get('num_recommendations')
        if kargs.get('numtopics') is not None:
            numtopics = kargs.get('numtopics')
        if kargs.get('numiterations') is not None:
            numiterations = kargs.get('numiterations')
        if kargs.get('batchsize') is not None:
            numiterations = kargs.get('batchsize')
        if kargs.get('learningrate') is not None:
            learningrate = kargs.get('learningrate')
        if kargs.get('learner') is not None:
            learner = kargs.get('learner')
        self.dataformat = "bag_of_word_in_memory"
        super().__init__(num_recommendations=num_recommendations, dataformat=self.dataformat)
        self.__itemuser = None
        self.__useritem = None
        self.__recommender1 = None
        self.__recommender2 = None
        self.__useridx2id = {}
        self.__userid2idx = {}
        self.__itemidx2id = {}
        self.__itemid2idx = {}
        self.__useridx2nvitems = {}
        self.__numtopics = numtopics
        self.__batchsize = batchsize
        self.__numiterations = numiterations
        self.__layers = [self.__numtopics*8, self.__numtopics*4, self.__numtopics*2, self.__numtopics]
        self.__learningrate = learningrate
        self.__learner = learner

    @staticmethod
    def __init_normal(shape, name=None):
        return initializations.normal(shape, scale=0.01, name=name)

    def __get_model(self, num_users, num_items, mf_dim=10, reg_layers=None, reg_mf=0):
        if reg_layers is None:
            reg_layers = [0, ] * len(self.__layers)
        assert len(self.__layers) == len(reg_layers)
        num_layer = len(self.__layers)  # Number of self.__layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                      init=self.__init_normal, W_regularizer=l2(reg_mf), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                      init=self.__init_normal, W_regularizer=l2(reg_mf), input_length=1)

        MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=self.__layers[0] / 2, name="mlp_embedding_user",
                                       init=self.__init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=self.__layers[0] / 2, name='mlp_embedding_item',
                                       init=self.__init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)

        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply

        # MLP part
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))

        mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
        for idx in range(1, num_layer):
            layer = Dense(self.__layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu',
                          name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        # Concatenate MF and MLP parts
        # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
        # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
        predict_vector = merge([mf_vector, mlp_vector], mode='concat')

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(predict_vector)
        model = Model(input=[user_input, item_input],
                      output=prediction)
        return model

    def update_model(self):
        """
        A method to update the recommender model.
        """
        temp_uservisits = self._uservisits
        unique_users = temp_uservisits['userid'].unique()
        for itemidx, itemid in self._allitems.iterrows():
            self.__itemid2idx[str(itemid['itemid'])] = itemidx
            self.__itemidx2id[itemidx] = str(itemid['itemid'])
        for useridx, userid in enumerate(unique_users):
            self.__userid2idx[str(userid)] = useridx
            self.__useridx2id[useridx] = str(userid)
            useritem = set(temp_uservisits[temp_uservisits['userid'] == userid]['itemid'].astype('str').values)
            allitem = set(self.__itemid2idx.keys())
            itemsnotinuser = allitem - useritem
            self.__useridx2nvitems[useridx] = list(itemsnotinuser)
            temp = pandas.DataFrame([{"userid": userid, "itemid": t, "rating": 0, "timestamp": "NA"} for t in itemsnotinuser])
            temp_uservisits = pandas.concat([temp_uservisits, temp])
        userid = temp_uservisits['userid'].values
        itemid = temp_uservisits['itemid'].values
        rating = temp_uservisits['rating'].values
        useridx = [self.__userid2idx[str(int(uid))] for uid in userid]
        itemidx = [self.__itemid2idx[str(int(iid))] for iid in itemid]
        model = self.__get_model(num_users=len(temp_uservisits['userid'].unique()), num_items=len(self._allitems),
                                 mf_dim=self.__numtopics)
        if self.__learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=self.__learningrate), loss='binary_crossentropy')
        elif self.__learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=self.__learningrate), loss='binary_crossentropy')
        elif self.__learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=self.__learningrate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=self.__learningrate), loss='binary_crossentropy')

        for epoch in range(self.__numiterations):
            t1 = time.time()
            hist = model.fit([numpy.array(useridx), numpy.array(itemidx)], numpy.array(rating),
                             batch_size=self.__batchsize, nb_epoch=1, verbose=0, shuffle=True)
            t2 = time.time()
        self.__recommender1 = model
        rating = list(map(numpy.double, rating))
        self.__itemuser = csr_matrix((rating, (useridx, itemidx)), shape=(len(set(useridx)), len(set(itemidx))))
        self.__recommender2 = implicit.als.AlternatingLeastSquares(factors=self.__numtopics)
        self.__recommender2.fit(self.__itemuser)

    def start_recommender(self, numRecs=None):
        """
        A method to start the recommender
        :param numRecs: number of recommendations
        """
        if numRecs is not None:
            self._num_recommendations = numRecs
        self.__useritem = self.__itemuser.T.tocsr()

    def __insertandretrieve_recommendation(self, userid=None, itemid=None):
        """
        A method to insert and retrieve recommendation.
        :param userid: the userid
        :param itemid: the itemid
        :return: the recommendations
        """
        def recommendation2rec(recommendationsip=None):
            recs = []
            for recommendation in recommendationsip:
                recs.append(self.__itemidx2id[recommendation[0]])
            return recs
        userid = str(userid)
        itemid = str(itemid)
        if userid in list(self.__userid2idx.keys()):
            useridx = self.__userid2idx[userid]
            userarray = numpy.asarray([useridx, ] * len(self.__itemidx2id.keys()))
            itemarray = numpy.asarray(list(self.__itemidx2id.keys()))
            predicted_ratings = self.__recommender1.predict([userarray, itemarray], batch_size=10, verbose=0)
            item_rating = {}
            for item, pr in zip(itemarray, predicted_ratings):
                item_rating[item] = pr[0]
            recommendations = sorted(item_rating.items(), key=lambda value: value[1], reverse=True)[:self._num_recommendations]
            recommendations = recommendation2rec(recommendationsip=recommendations)
        else:
            if itemid in list(self.__itemid2idx.keys()):
                itemidx = self.__itemid2idx[itemid]
                recommendations = self.__recommender2.similar_items(itemidx, N=self._num_recommendations)
                recommendations = recommendation2rec(recommendationsip=recommendations)
            else:
                recommendations = list(self.__itemid2idx.keys())
                random.shuffle(recommendations)
                recommendations = recommendations[:self._num_recommendations]
        return recommendations

    def insertandretrieve_recommendation_fromlog(self, userid=None, itemid=None):
        return self.__insertandretrieve_recommendation(userid=userid, itemid=itemid)

    def insertandretrieve_recommendation_fromcc(self, userid=None, itemid=None):
        return self.__insertandretrieve_recommendation(userid=userid, itemid=itemid)

    def stop_recommender(self):
        self.__useritem = csr_matrix((0, 0))
