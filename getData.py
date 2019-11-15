import os

########################################################################################################################

def GetPatients(directoryPath):
    """
    A function to get the paths to the different patients from a directory.
    :param directoryPath: a string with the path to the directory with the patients.
    :return: a list with the paths to the patients.
    """
    patientPaths = []
    patientNames = []
    for entry in os.listdir(directoryPath):
        patientPath = os.path.join(directoryPath, entry)
        if os.path.isdir(patientPath):
            patientPaths.append(patientPath)
            patientNames.append(entry)
    return patientPaths, patientNames


