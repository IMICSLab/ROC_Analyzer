#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Aug  2023

@author: Ernest Namdar
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import copy
matplotlib.use('Qt5Agg')


class KNCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(KNCanvas, self).__init__(fig)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMICS ROC Analyzer")
        self.setWindowIcon(QIcon('icons/iMICSAnalyzer.ico'))
        self.setGeometry(450,150,1350,1050)
        self.UI()
        self.show()

    def UI(self):
        self.toolBar()
        self.widgets()
        self.layouts()

    def toolBar(self):
        #########################Main Menu; nested in the top layout###############
        self.menubar=self.menuBar()
        file=self.menubar.addMenu("File")
        mode=self.menubar.addMenu("Mode")
        ROCopts=self.menubar.addMenu("ROC Options")
        helpMenu=self.menubar.addMenu("Help")
        ###########################Sub Menu Items################
        ######File: Open CSV & Exit########
        open=QAction("Open CSV",self)
        open.setShortcut("Ctrl+O")
        open.triggered.connect(self.importCSVFunc)
        file.addAction(open)
        exit=QAction("Exit",self)
        exit.setIcon((QIcon("icons/exit2.png")))
        exit.triggered.connect(self.exitFunc)
        file.addAction(exit)
        ######Mode: Investigation########
        self.investigationMode=QAction("Investigation",self, checkable=True)
        self.investigationMode.triggered.connect(self.getMode)
        mode.addAction(self.investigationMode)
        ######ROCopts: J point########
        self.JpointOption=QAction("Youden's J",self, checkable=True)
        self.JpointOption.triggered.connect(self.getJoption)
        ROCopts.addAction(self.JpointOption)
        self.Jopt = False
        ######Help: Help########
        HelpWindow=QAction("Help",self)
        HelpWindow.triggered.connect(self.helpFunc)
        helpMenu.addAction(HelpWindow)

    def widgets(self):
        self.central=QTabWidget()
        self.setCentralWidget(self.central)
        #############Data Table##################
        self.DataTable = QTableWidget()
        self.DataTable.setColumnCount(4)
        self.DataTable.setHorizontalHeaderItem(0,QTableWidgetItem("Include"))
        self.DataTable.setHorizontalHeaderItem(1,QTableWidgetItem("Patient ID"))
        self.DataTable.setHorizontalHeaderItem(2,QTableWidgetItem("Prediction"))
        self.DataTable.setHorizontalHeaderItem(3,QTableWidgetItem("Ground Truth"))
        self.DataTable.setRowCount(50)
        self.DataTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ###############ROC#################
        self.ROC = KNCanvas(self, width=5, height=4, dpi=100)
        self.ROC.axes.plot([0, 1], [0, 1],'r--')
        self.ROC.axes.set_ylabel('True Positive Rate')
        self.ROC.axes.set_xlabel('False Positive Rate')
        self.ROC.axes.set_title('Receiver Operating Characteristic')
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.tlbar = NavigationToolbar(self.ROC, self)
        self.drawButton=QPushButton("Draw ROC",self)
        self.drawButton.clicked.connect(self.drawROC)
        ###############Image#################
        self.Img =QLabel(self)
        self.Img.setPixmap(QPixmap('images/default.png').scaledToWidth(self.ROC.width()))#)600))
        #############Outliers##################
        self.combo = QComboBox(self)
        self.combo.addItems(["Actual Positives","Actual Negatives","Overall"])
        self.OutliersTable = QTableWidget()
        self.OutliersTable.setColumnCount(2)
        self.OutliersTable.setHorizontalHeaderItem(0,QTableWidgetItem("Patient ID"))
        self.OutliersTable.setHorizontalHeaderItem(1,QTableWidgetItem("Score"))
        self.OutliersTable.setRowCount(50)
        self.calcbutton=QPushButton("Calculate Oulier Scores",self)
        self.calcbutton.clicked.connect(self.calcOutliers)
        ###############AUC_Loss#################
        self.AUCLoss = KNCanvas(self, width=5, height=4, dpi=100)
        self.AUCLoss.axes.set_ylabel('True Positive Rate')
        self.AUCLoss.axes.set_xlabel('False Positive Rate')
        self.AUCLoss.axes.set_title('Receiver Operating Characteristic')
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.tlbar2 = NavigationToolbar(self.AUCLoss, self)
        self.aucvizButton=QPushButton("Visualize AUC Loss",self)
        self.aucvizButton.clicked.connect(self.aucviz)

    def layouts(self):
        ##########################Creating Layouts###############
        self.mainCentralLayout=QHBoxLayout()  # the main layout hosts two sections table&vis
        self.tableLayout= QVBoxLayout()
        self.outlierLayout= QVBoxLayout()
        self.visLayout= QVBoxLayout()
        #########################################################

        #################Add widgets###################
        self.tableLayout.addWidget(self.DataTable)
        self.outlierLayout.addWidget(self.tlbar2)
        self.outlierLayout.addWidget(self.AUCLoss)
        self.outlierLayout.addWidget(self.aucvizButton)
        self.outlierLayout.addWidget(self.combo)
        self.outlierLayout.addWidget(self.OutliersTable)
        self.outlierLayout.addWidget(self.calcbutton)
        self.visLayout.addWidget(self.Img)
        self.visLayout.addWidget(self.tlbar)
        self.visLayout.addWidget(self.ROC)
        self.visLayout.addWidget(self.drawButton)
        ###############################################

        ################Nest Layouts###################
        self.mainCentralLayout.addLayout(self.tableLayout)
        self.mainCentralLayout.addLayout(self.outlierLayout)
        self.mainCentralLayout.addLayout(self.visLayout)
        self.central.setLayout(self.mainCentralLayout)
        ###############################################

    def exitFunc(self):
        mbox=QMessageBox.information(self,"Warning","Are you sure you want to exit?",
                                     QMessageBox.Yes|QMessageBox.No,
                                     QMessageBox.No)
        if mbox==QMessageBox.Yes:
            sys.exit(0)

    def importCSVFunc(self):
        path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv(__file__), 'CSV(*.csv)')
        if path[0] != '':
            self.dframe = pd.read_csv(path[0])
            col_headers = list(self.dframe)
            self.DataTable.setRowCount(self.dframe.shape[0])
            for i in range(self.dframe.shape[0]):
                self.DataTable.setItem(i,0,QTableWidgetItem(str(self.dframe["Include"][i])))
                self.DataTable.setItem(i,1,QTableWidgetItem(str(self.dframe["Patient ID"][i])))
                self.DataTable.setItem(i,2,QTableWidgetItem(str(self.dframe["Prediction"][i])))
                self.DataTable.setItem(i,3,QTableWidgetItem(str(self.dframe["Ground Truth"][i])))
            # print(self.dframe)

    def helpFunc(self):
        mbox=QMessageBox.information(self,"Help","See our website at http://www.imics.ca or contact ernest.namdar@utoronto.ca")

    def drawROC(self):
        self.getMode()
        self.getPredictions()
        self.getLabels()
        fpr, tpr, threshold = metrics.roc_curve(self.Lb, self.Pr)
        diff = 0
        Jtpr = 0
        Jfpr= 0
        for i in range(len(fpr)):
            if (tpr[i]-fpr[i])>diff:
                diff = (tpr[i]-fpr[i])
                Jtpr = tpr[i]
                Jfpr = fpr[i]
        self.Jtpr = Jtpr
        self.Jfpr = Jfpr
        self.update_plot(fpr, tpr)
        self.ROC.show()

    def aucviz(self):
        self.getMode()
        self.getPredictions()
        self.getLabels()
        fpr, tpr, threshold = metrics.roc_curve(self.Lb, self.Pr)
        self.update_plot2(fpr, tpr)
        self.AUCLoss.show()

    def calcOutliers(self):
        self.getPredictions()
        self.getLabels()
        self.getpIDs()
        Nap=sum(self.Lb)
        # print("Nap", Nap)
        Nan=len(self.Lb)-Nap
        # print("Nan", Nan)
        df = pd.DataFrame({'pID':self.pIDs,
                           'Prediction': self.Pr,
                           'Label': self.Lb})
        df = df.sort_values("Prediction", ascending=True)
        df = df.reset_index(drop=True)
        df.insert(3, "Effect", "0")
        df.insert(4, "Severity", "n")
        df.insert(5, "Score", "n")
        df.insert(6, "Rank", "n")
        rec_area = (1/Nap)*(1/Nan)
        for i in range(Nan):
            df["Severity"][i]=Nan-i
            if df["Label"][i]==0:
                df["Effect"][i]='p'
            elif df["Label"][i]==1:
                df["Effect"][i]='n'

        for i in range(Nap):
            df["Severity"][df.shape[0]-i-1]=Nap-i
            if df["Label"][df.shape[0]-i-1]==1:
                df["Effect"][df.shape[0]-i-1]='p'
            elif df["Label"][df.shape[0]-i-1]==0:
                df["Effect"][df.shape[0]-i-1]='n'

        for i in range(df.shape[0]):
            if df["Effect"][i]=='p':
                df["Score"][i]=0
            elif df["Effect"][i]=='n':
                if df["Label"][i]==0:
                    df["Score"][i]=df["Severity"][i]*(1/Nap)
                elif df["Label"][i]==1:
                    df["Score"][i]=df["Severity"][i]*(1/Nan)
                else:
                    print("Error in Labels (neither 0 nor 1)")
            else:
                print("Error om Effect (neither n nor p)")

        ######Separate Actual Psitives
        df_p = copy.deepcopy(df)
        drop_inds = []
        for i in range(df_p.shape[0]):
            if df_p["Label"][i]==0:
                drop_inds.append(i)
        df_p = df_p.drop(drop_inds)
        df_p = df_p.sort_values("Score", ascending=False)
        df_p = df_p.reset_index(drop=True)
        for i in range(df_p.shape[0]):
            df_p["Rank"][i] = df_p.shape[0]-i
        drop_inds = []
        for i in range(df_p.shape[0]):
            if df_p["Score"][i]==0:
                drop_inds.append(i)
        df_p = df_p.drop(drop_inds)
        df_p = df_p.sort_values("Score", ascending=False)
        df_p = df_p.reset_index(drop=True)
        # print(df_p)

        ######Separate Actual Negatives
        df_n = copy.deepcopy(df)
        drop_inds = []
        for i in range(df_n.shape[0]):
            if df_n["Label"][i]==1:
                drop_inds.append(i)
        df_n = df_n.drop(drop_inds)
        df_n = df_n.sort_values("Score", ascending=False)
        df_n = df_n.reset_index(drop=True)
        for i in range(df_n.shape[0]):
            df_n["Rank"][i] = df_n.shape[0]-i
        drop_inds = []
        for i in range(df_n.shape[0]):
            if df_n["Score"][i]==0:
                drop_inds.append(i)
        df_n = df_n.drop(drop_inds)
        df_n = df_n.sort_values("Score", ascending=False)
        df_n = df_n.reset_index(drop=True)
        # print(df_n)

        drop_inds = []
        for i in range(df.shape[0]):
            if df["Score"][i]==0:
                drop_inds.append(i)
        df = df.drop(drop_inds)
        df = df.sort_values("Score", ascending=False)
        df = df.reset_index(drop=True)
        for i in range(df.shape[0]):
            if df["Label"][i]==1:
                for j in range(df_p.shape[0]):
                    if df["pID"][i]==df_p["pID"][j]:
                        df["Rank"][i]=df_p["Rank"][j]
                        break
            elif df["Label"][i]==0:
                for j in range(df_n.shape[0]):
                    if df["pID"][i]==df_n["pID"][j]:
                        df["Rank"][i]=df_n["Rank"][j]
                        break
        # print(df)

        cohort=self.combo.currentText()
        # print(cohort)
        if cohort == "Actual Positives":
            self.OutliersTable.setRowCount(df_p.shape[0])
            for i in range(df_p.shape[0]):
                self.OutliersTable.setItem(i,0,QTableWidgetItem(str(df_p["pID"][i])))
                self.OutliersTable.setItem(i,1,QTableWidgetItem(str(df_p["Score"][i])))
            self.ScoreTable=df_p
        elif cohort == "Actual Negatives":
            self.OutliersTable.setRowCount(df_n.shape[0])
            for i in range(df_n.shape[0]):
                self.OutliersTable.setItem(i,0,QTableWidgetItem(str(df_n["pID"][i])))
                self.OutliersTable.setItem(i,1,QTableWidgetItem(str(df_n["Score"][i])))
            self.ScoreTable=df_n
        elif cohort == "Overall":
            self.OutliersTable.setRowCount(df.shape[0])
            for i in range(df.shape[0]):
                self.OutliersTable.setItem(i,0,QTableWidgetItem(str(df["pID"][i])))
                self.OutliersTable.setItem(i,1,QTableWidgetItem(str(df["Score"][i])))
            self.ScoreTable=df
        # print(df["Severity"])

    def getPredictions(self):
        self.Pr = []
        for i in range(self.DataTable.rowCount()):
            if int(self.DataTable.item(i,0).text()) == 1:
                self.Pr.append(float(self.DataTable.item(i,2).text()))
        #print(self.Pr)

    def getLabels(self):
        self.Lb = []
        for i in range(self.DataTable.rowCount()):
            if int(self.DataTable.item(i,0).text()) == 1:
                self.Lb.append(int(self.DataTable.item(i,3).text()))
        #print(self.Lb)

    def getpIDs(self):
        self.pIDs = []
        for i in range(self.DataTable.rowCount()):
            if int(self.DataTable.item(i,0).text()) == 1:
                self.pIDs.append(str(self.DataTable.item(i,1).text()))
        #print(self.Lb)

    def update_plot(self, X, Y):
        self.roc_auc = metrics.auc(X, Y)
        self.ROC.axes.cla()  # Clear the canvas.
        self.ROC.axes.plot(X, Y, 'b', label = 'AUC = %0.2f' % self.roc_auc)
        if self.Jopt is True:
            self.ROC.axes.plot(self.Jfpr, self.Jtpr, 'r', marker = '*', label = "Youden's J")
        self.ROC.axes.set_ylabel('True Positive Rate')
        self.ROC.axes.set_xlabel('False Positive Rate')
        self.ROC.axes.set_title('Receiver Operating Characteristic')
        self.ROC.axes.legend(loc = 'lower right')
        self.ROC.axes.plot([0, 1], [0, 1],'r--')
        self.ROC.axes.set_xlim([0, 1])
        self.ROC.axes.set_ylim([0, 1])
        # Trigger the canvas to update and redraw.
        self.ROC.draw()

    def update_plot2(self, X, Y):
        self.AUCLoss.axes.cla()  # Clear the canvas.
        self.AUCLoss.axes.plot(X, Y, 'b')
        self.AUCLoss.axes.set_ylabel('True Positive Rate')
        self.AUCLoss.axes.set_xlabel('False Positive Rate')
        self.AUCLoss.axes.set_title('Receiver Operating Characteristic')
        self.AUCLoss.axes.set_xlim([0, 1])
        self.AUCLoss.axes.set_ylim([0, 1])

        self.getPredictions()
        self.getLabels()
        self.getpIDs()
        Nap=sum(self.Lb)
        Nan=len(self.Lb)-Nap

        self.AUCLoss.axes.set_xticks(np.arange(0,1,1/Nan))
        self.AUCLoss.axes.set_yticks(np.arange(0,1,1/Nap))
        self.AUCLoss.axes.grid(which='both')
        self.AUCLoss.axes.set_xticklabels([])
        self.AUCLoss.axes.set_yticklabels([])
        self.drawLostAUC()
        # Trigger the canvas to update and redraw.
        self.AUCLoss.draw()

    def getOutlierSelection(self):
        for item in self.OutliersTable.selectedItems():
            return int(item.row())
            break
            #print(item.text(),item.row(),item.column())

    def drawLostAUC(self):
        ind = self.getOutlierSelection()
        if ind is None:
            ind = 0
        self.getPredictions()
        self.getLabels()
        self.getpIDs()
        Nap=sum(self.Lb)
        Nan=len(self.Lb)-Nap
        rank = self.ScoreTable["Rank"][ind]
        lb = self.ScoreTable["Label"][ind]
        # print(lb, rank)
        if lb == 1:
            # print("Nap", Nap)
            # print("Nan", Nan)
            # print("rank", rank)
            # print("Lb", lb)
            # print("1/Nap", 1/Nap)
            y = (rank/Nap)-(1/Nap)
            x = 0
            # print("y:", y)
            rect = patches.Rectangle((x,y),1,1/Nap,linewidth=1,color='r',alpha=0.6, fill=True)
        elif lb ==0:
            y=0
            x=1-(rank/Nan)
            rect = patches.Rectangle((x,y),1/Nan,1,linewidth=1,color='r',alpha=0.6, fill=True)
        else:
            print("Error in label (neither 0 nor 1)")
        self.AUCLoss.axes.add_patch(rect)

    def getMode(self):
        if self.investigationMode.isChecked():
            self.DataTable.setEditTriggers(QAbstractItemView.AllEditTriggers)
            # print("We are in the Investigation Mode")
        if not self.investigationMode.isChecked():
            self.DataTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
            # print("We are NOT in the Investigation Mode")

    def getJoption(self):
        if self.JpointOption.isChecked():
            self.Jopt = True
            # print("We are in the Investigation Mode")
        if not self.JpointOption.isChecked():
            self.Jopt = False
            # print("We are NOT in the Investigation Mode")


def main():
    App=QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec_())

if __name__=='__main__':
    main()
