File = 'example.xlsx'
def write_xlsx(file_name,top_title,data_list):
    style = xw.easyxf()#字体风格设置，此处为默认值
    oldwb = xr.open_workbook(file_name)#打开工作簿
    newwb = xl_copy(oldwb)#复制出一份新工作簿
    newws = newwb.get_sheet(0)#获取指定工作表，0表示实际第一张工作表

    column = 1  #在excel表格的第column行开始写数据
    newws.write(0,column-1,top_title)
    for i in range(len(data_list)):
        newws.write(i+1, column-1, data_list[i],style) #把列表a中的元素逐个写入第一列，0表示实际第1列,i+1表示实际第i+2行
    newwb.save(file_name)#保存修改

write_xlsx(File,'LOSS',LOSS_VALUE_total)
