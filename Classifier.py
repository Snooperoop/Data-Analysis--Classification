import numpy as np
import pandas as pd


class Regularized_SVM:
    """This SVM Only outputs a linear divider, meaning there are no degrees to mess around with."""
    def __init__(self,Features:list[str], Target:str):
        self.Features = Features
        self.Target = Target
        self.Weights = np.zeros((len(self.Features)))
        self.Bias = 0
        pass

    def Train_Model(self, Data_Frame, Epochs, Learning_Rate, Lambda):
        """
        Your Target Columns must be of the values -1 and 1
        """
        for iterations in range(Epochs):
            for row in range(len(Data_Frame)):    
                Y_i = Data_Frame[self.Target].iloc[row]
                X_i = Data_Frame[self.Features].iloc[row].to_numpy()

                DR_Weights = None
                DR_Bias = None
                margin = Y_i * ( self.Weights @ X_i + self.Bias)
                if margin >= 1:    
                    DR_Weights = Lambda * self.Weights
                    DR_Bias = 0
                else: 
                    DR_Weights =  Lambda * self.Weights - X_i * Y_i
                    DR_Bias = - Y_i
                self.Weights = self.Weights - Learning_Rate * DR_Weights
                self.Bias = self.Bias - Learning_Rate * DR_Bias

    def Predict(self, X_i):
        """Returns an approximation"""
        return self.Weights @ X_i + self.Bias

    def Predict_Class(self, X_i):
        val = self.Predict(X_i)
        # np.sign is a function that returns -1, 0, or 1 based on a values sign.
        return np.sign(val)



    def Test_Model(self, Data_Frame):
        """This model tests for classification accuracy, not an overall judgement"""
        Right = 0
        Total = len(Data_Frame)
        for row in range(len(Data_Frame)):
            Y_i = Data_Frame[self.Target].iloc[row]
            X_i = Data_Frame[self.Features].iloc[row].to_numpy()
            if self.Predict_Class(X_i) == Y_i:
                Right+=1
        Accuracy = Right / Total
        return Accuracy


class Logistic_Regression:
    def __init__(self, Features: list[str], Target:str):
        self.Features = Features
        self.Target = Target
        self.Weights = np.zeros((len(Features) + 1, 1))
        pass

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def Train_Model(self, Data_Frame, Epochs, Learning_Rate, tol=1e-7):
        # Declares and shapes the numpy matrix of Inputs (numb of rows x numb of cols + 1) 
        Inputs = Data_Frame[self.Features].to_numpy()
        number_of_rows, number_of_columns = Inputs.shape
        Input_Matrix = np.c_[np.ones((number_of_rows, 1)), Inputs]
        # Declares and shapes the numpy matrix of Outputs (numb of rows x 1)  
        Output_Vector = np.array(Data_Frame[self.Target]).reshape(number_of_rows,1)
        for iterations in range(Epochs):
            # GRADIENT DESCENT
            DR_Weights = np.zeros_like(self.Weights)
            DR_Weights = (1/number_of_rows) *  np.transpose(Input_Matrix) @ (self.sigmoid_function(Input_Matrix @ self.Weights) - Output_Vector)
            self.Weights = self.Weights - Learning_Rate * DR_Weights    
        # A tolerance check adds an early break for when an algorithm is getting closer to its potential optimized point
            if np.linalg.norm(DR_Weights) < tol:
                break

        return 

    def Predict(self, Inputs):
        """
        Takes an input to then give a probability of how likely a parameter(s) is to its target
        """
        Inputs = np.array(Inputs)
        # Checks to see if shapes are matching for matrix multiplication
        Input_w_bias = np.c_[np.ones((Inputs.shape[0], 1)), Inputs]
        mx,nx = Input_w_bias.shape
        mw,nw = self.Weights.shape
        if nx != mw:
            raise ValueError("INPUT DIMENSIONS CANNOT FIT WEIGHT \nDIMENSIONS: X:"+str(mx)+" x "+str(nx)+"\nDIMENSIONS WEIGHTS: " + str(mw) + " x " +str(nw))
        return self.sigmoid_function( Input_w_bias @ self.Weights)
    def Predict_Class(self, Inputs, Divider: float = 0.5):
        return self.Predict(Inputs) >= Divider
    
    def Test_Class(self, Data_Frame, Divider=0.5):
        """
        Only use to judge classification accuracy, not OVERALL accuracy
        """
        Accuracy = 0
        Right_Classifications = 0
        Total_Classifications = len(Data_Frame)
        for row in range(Total_Classifications):
            real = Data_Frame[self.Target].iloc[row]
            Inputs = Data_Frame[self.Features].iloc[row].to_numpy().reshape(1, -1)
            prediction = self.Predict_Class(Inputs, Divider)
            if int(real) == prediction:
                Right_Classifications+=1
        Accuracy = Right_Classifications / Total_Classifications
        return Accuracy
    