#!/usr/bin/env python


from email.mime import audio
from sys import flags
from turtle import up
from unittest import result
import rospy
import numpy as np
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from audio_common_msgs.msg import Audio_Result
from tkinter import *
rospy.init_node('command_node', anonymous=True)
command_pub = rospy.Publisher('/audio_recognition_msg', String ,queue_size = 5)
audio_pub = rospy.Publisher('/audio',AudioData,queue_size=5)
command = String()
audio_data = AudioData()

def callback(msg):
    emotion = msg.emotion
    confidence_rate =msg.confidence_rate
    return emotion, confidence_rate


def event():
    button['text'] = 'recording...'
    command.data = 'on'
    command_pub.publish(command)
def event2():
    button2['text'] = 'predict...'
    command.data = 'end'
    command_pub.publish(command)
    audio_pub.publish(audio_data)
    result_1 = rospy.wait_for_message('/audio_recognition_result',Audio_Result)
    label1.configure(text='Emotion=%s' % result_1.emotion)
    label2.configure(text='Probability=%s' % result_1.confidence_rate)
def event3():
    button['text'] = 'start'
    button2['text'] = 'end'
    command.data = 'reset'
    command_pub.publish(command)

# def command_function(command):
tk = Tk()
tk.title('Speech Emotion Recognizier')
tk.geometry('325x120')
label1=Label(tk,width=30, height=2,font=('맑은 고딕',12,'bold'),bg='#2F5597',fg='white')
label2=Label(tk,width=30, height=2, font=('맑은 고딕',12,'bold'),bg='#2F5597',fg='white')
button = Button(tk, text='start',width = 10,height=1,command=event)
button2 = Button(tk, text='end',width = 10,height=1, command = event2)
button3 = Button(tk , text='reset',width = 10,height=1, command = event3 )
button.grid(row=0,column=0,padx = 5, pady =10)
button2.grid(row=0,column=1, padx = 5, pady =10)
button3.grid(row=0, column=2, padx= 5, pady=10)
label1.grid(row=1,columnspan=3, padx=5,)
label2.grid(row=2, columnspan=3, padx=5,)

tk.mainloop()
