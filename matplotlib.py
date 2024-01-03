#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_excel('C:\\Users\\user\\OneDrive\\Documents\\EXCEL AKKA LESSON\\Book1.xlsx')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df['TELUGU'].median()


# In[8]:


df.groupby('TELUGU')['S NO'].sum()


# In[9]:


df.loc[(df['TELUGU']>50)&(df['TELUGU']<80)].sort_values


# In[10]:


df.loc[(df['TELUGU']>50)&(df['TELUGU']<80)].min()


# In[11]:


x=[1,2,3,4,5]
y=[20,90,43,19,5]
plt.plot(x,y,marker='.',color='red',ms=14)
plt.title('Chaithu')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()


# In[12]:


x=[5,10,15,20,25]
y=[5,45,23,76,30]
plt.plot(x,y,marker='*',color='Green',mfc='r',mec='y')
f1={'family':'cambria','color':'yellow','size':30}
plt.title('chaithu',fontdict=f1)
plt.show()


# In[13]:


x=[1,2,3,4,5]
y=[15,65,24,34,10]
plt.plot(x,y,marker='*',ls='None',ms=15,mfc='r',mec='b')
f1={'family':'cambria','color':'red','size':20}
plt.title("Sanju",fontdict=f1)
plt.xlabel('a-axis',fontdict=f1)
plt.ylabel('b-axis',fontdict=f1)
plt.grid(axis='y')
plt.show()


# In[14]:


x=[2,4,6,8,10,12]
y=[10,28,36,43,23,15]
plt.plot(x,y,marker='*',color="Green",mec='r',ls='--',lw=2)
f1={'family':'calibri','color':'red','size':16}
f2={'family':'cambria','color':'Black','size':20}
plt.title('Matplotlib',fontdict=f1)
plt.xlabel('Stu',fontdict=f2)
plt.ylabel('Marks',fontdict=f2)
plt.grid(axis='x',ls=':',color='r',lw=1)
plt.show()


# In[15]:


x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.show()


# In[16]:


a=[1,2,3,4,5]
b=[10,20,35,16,28]
c=[20,76,34,7,14]
plt.subplot(2,1,1)
plt.plot(a,b)
plt.subplot(2,1,1)
plt.plot(a,c)
plt.show()


# In[17]:


a=[10,20,30,40,50]
b=[100,200,300,400,500]
c=[50,100,150,200,250]
f1={'family':'calibri','color':'red','size':20}
f2={'family':'cambria','color':'yellow','size':10}
f3={'family':'candara','color':'black','size':10}
plt.suptitle('Python',fontdict=f1)
plt.subplot(1,2,1)
plt.plot(a,b)
plt.title('Data',fontdict=f2)
plt.subplot(1,2,2)
plt.plot(a,c)
plt.title('Type',fontdict=f3)
plt.show()


# In[18]:


a=[10,20,30,40,50,60]
b=[20,24,34,54,65,10]
c=[30,31,24,54,18,43]
f1={'family':'candara','color':'Red','size':20}
f2={'family':'cambria','color':'Yellow','size':10}
plt.suptitle('Matplotlib',fontdict=f1)
plt.subplot(1,2,1)
plt.plot(a,b,marker='*')
plt.subplot(1,2,1)
plt.plot(a,c,marker=".")
plt.xlabel('Std',fontdict=f1)
plt.ylabel('Markes',fontdict=f2)
plt.legend(['Year','Months'],loc=2)
plt.show()


# In[19]:


x=[1,2,3,4,5,6]
y=[10,20,30,42,22,15]
plt.plot(x,y)
plt.legend(['raju'],loc=1,framealpha=1,facecolor='red',shadow=True)
plt.show()


# In[20]:


x=[1,2,3,4,5,6]
y=[10,20,30,40,50,10]
plt.scatter(x,y,color='r',ls='-:')
f1={'family':'cambria','color':'Blue'}
plt.title('Scatter',fontdict=f1)
plt.xlabel('Stu')
plt.ylabel('Markes')
plt.grid(axis='x',c='yellow')
plt.legend(['Raju'],loc=2,shadow=False)
plt.show()


# In[21]:


df.head()


# In[22]:


df1=df.groupby('NAME')['TOTAL'].sum()


# In[23]:


df1


# In[24]:


df1.plot( kind='pie',x='NAME',y='TOTAL')


# In[25]:


df1.plot(x='NAME',y='TOTAL',marker='*',color='red')


# In[26]:


df.plot(kind='bar',x='NAME')


# In[27]:


df.plot(kind='hist',x='NAME')


# In[28]:


df.head()


# In[ ]:




