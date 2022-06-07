import tkinter as tk


class WindowClient:
  """
    GUI client offering an input/output interface for text analysys
  """

  def __init__(self, engine):
    """
      Creates a new instance of a tkinter window; takes an analyzer object
      with .setup() for env. initialization method and .predict() giving
      values resut in a tuple
    """
    self.__engine = engine
    self.__background_color = 'black'
    self.__system_color = '#9a5aed'


  def launch(self):
    """
      Environment setup
    """
    self.__engine.setup()
    # window params
    self.__root = tk.Tk()
    self.__root.resizable(False, False)
    self.__root.title('Sentiment analyzer')
    self.__root.geometry('1000x600+400+300')
    self.__root.config(bg = self.__background_color)
    # initializing entries
    self.__add_result_label()
    self.__add_input_label('Enter some text:')
    self.__add_textarea()
    self.__add_predict_btn()
    self.__add_clear_btn()
    # opening window
    self.__root.mainloop()

  
  def __add_input_label(self, text):
    label = tk.Label(self.__root, text=text, height=2, width=15,
      background=self.__background_color, foreground=self.__system_color,
      font=('Arial', 16, 'bold'))
    label.pack(padx=25, pady=5, anchor='nw')


  def __add_textarea(self):
    self.__textarea = tk.Text(self.__root, height=15, width=40,
      background='#1b1b1c', foreground=self.__system_color,
      font=('Arial', 18), borderwidth=2)
    self.__textarea.pack(padx=20, pady=1, anchor='nw')


  def __add_result_label(self):
    self.__result_label = tk.Label(self.__root, text='value:\nsentiment:', height=2, width=30,
      background=self.__background_color, foreground=self.__system_color,
      font=('Arial', 16, 'bold'), anchor='w')
    self.__result_label.pack(side='right')


  def __add_predict_btn(self):
    btn = tk.Button(self.__root, text='Predict', height=2, width=17,
      background='#1b1b1c', foreground=self.__system_color,
      font=('Arial', 16, 'bold'), borderwidth=2, command=self.__predict_action)
    btn.pack(padx=27, pady=5, side='left')


  def __add_clear_btn(self):
    btn = tk.Button(self.__root, text='Clear', height=2, width=17,
      background='#1b1b1c', foreground=self.__system_color,
      font=('Arial', 16, 'bold'), borderwidth=2, command=self.__clear_input)
    btn.pack(padx=20, pady=5, side='left')

  
  def __clear_input(self):
    self.__textarea.delete('1.0', 'end')
    self.__set_result_value()

  
  def __predict_action(self):
    text = self.__textarea.get('1.0', 'end')[:-1]
    if not text: return
    result = self.__engine.predict(text)
    self.__set_result_value(result)


  def __set_result_value(self, value=None):
    if value is None:
      self.__result_label.config(text='value:\nsentiment:')
    else:
      numeric, string = value
      text = 'value: %.3f\nsentiment: %s' % (numeric, string)
      self.__result_label.config(text=text)
