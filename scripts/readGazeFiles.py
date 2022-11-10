from praatio import textgrid
tg = textgrid.openTextgrid('../data/gaze_files/DVA1A.gaze', False)
print(dir(tg))
print('\n\n\n')
print(tg.tierDict)
print('\n\n\n')
print(dir(tg.tierDict['kijkrichting spreker1 [v] (TIE1)']))
print('\n\n\n')
print(tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList[0])