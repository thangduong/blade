import framework.utils.common as utils

# wrap up thesaurus
class MobyThesaurus:
    def __init__(self, filename):
        self._synonyms = {}     # maps word to a list of synonyms in order of closeness
        with open(filename, 'r') as f:
            for line in f:
                words = line.rstrip().lstrip().split(',')
                self._synonyms[words[0]] = words[1:]
        print(self._synonyms)

    def get_synonyms(self, word):
        return utils.get_dict_value(self._synonyms, word, [])

if __name__ == "__main__":
    t = MobyThesaurus('C:\\digitalcortex\\mrcmodels.thang_sat\\data\\mobythes.aur')
