import pandas as pd
import matplotlib.pyplot as plt


class BigData():

    def __init__(self, base_dados=None, data_frame=None):
        self.base_dados = base_dados
        self.data_frame = data_frame

    def main(self, command):
        """ GERENCIA A APLICACAO """

        switcher = {
            "1": self.menu, 
            "2": self.create_df, 
            "3": self.show_types_df, 
            "4": self.create_df_media, 
            "5": self.show_metricas_df, 
            "6": self.order_df, 
            "7": self.valores_df, 
            "8": self.plotar_grafico, 
        }
        
        while command != '9':
            if switcher.get(command):
                switcher[command]()

            if command in ("", "1"):
                self.exibe_sub_menu()

            command = input(': ')

    def exibe_menu(self):
        """ IMPRIME O MENU """

        print("*=============================*")
        print("* SELECIONE UMA BASE DE DADOS *")
        print("*=============================*")
        print("* 1-EVOLUCAO DOS VEICULOS     *")
        print("* 2-RENDA MAIOR DE 50k ANO    *")
        print("* 3-FLOR IRIS                 *")
        print("* 4-VARIEDADES DE KECIMEN     *")
        print("* 5-ABALONE                   *")
        print("*=============================*")

    def exibe_sub_menu(self):
        """ IMPRIME O SUB-MENU """

        print("*=========================================*")
        print("*             MENU DE OPCOES              *")
        print("*=========================================*")
        print("* 1-ALTERAR BASE DE DADOS                 *")
        print("* 2-CRIAR DATA_FRAME                      *")
        print("* 3-MOSTRAR OS TIPOS DAS COLUNAS DO DF    *")
        print("* 4-PREENCHER DADOS AUSENTES COM A MEDIA  *")
        print("* 5-MOSTRAR METRICAS DO DATA_FRAME        *")
        print("* 6-ORDENAR DATA_FRAME EM ASC/DESC        *")
        print("* 7-MOSTRAR VALORES ABAIXO/ACIMA DA MEDIA *")
        print("* 8-PLOTAR GRAFICO                        *")
        print("* 9-FINALIZAR                             *")
        print("*=========================================*")

    def menu(self):
        """ EXIBE AS BASES DE DADOS DISPONIVEIS """

        self.exibe_menu()
        selected_bd = input(": ")

        dict_db = {
            "1": "car.data",
            "2": "adult.data",
            "3": "iris.data",
            "4": "raisin_dataset.data",
            "5": "abalone.data",
        }

        if dict_db.get(selected_bd):
            self.base_dados = dict_db[selected_bd]
            print("BASE DE DADOS SELECIONADA:", dict_db[selected_bd])

        else:
            print("OPCAO INVALIDA")
            self.menu()

    def create_df(self):
        """ CRIA DATA FRAME """

        df = pd.read_csv(f'./base_de_dados/{self.base_dados}')
        self.data_frame = df
        print("DF:\n", df)

    def create_df_media(self):
        """ CRIA UM DATA FRAME COM AS MEDIAS """

        if self.check_df():
            select_df = self.data_frame.select_dtypes(include=['float', 'int'])
            media = select_df.mean()

            print("INCLUINDO MEDIA DA COLUNA NOS VALORES VAZIOS>>>")
            self.data_frame[select_df.columns] = select_df.fillna(media)

            print("DF:\n", self.data_frame)

    def check_df(self):
        """ VERIFICA SE EXISTE UM DATA FRAME CRIADO """

        have_df = False if self.data_frame is None else True

        if not have_df:
            print("NENHUM DATA FRAME DISPONIVEL!")

        return have_df

    def show_types_df(self):
        """ EXIBE O TIPO DE CADA COLUNA DO DATA_FRAME """
        
        if self.check_df():
            print('TYPES:\n', self.data_frame.dtypes)

    def show_metricas_df(self):
        """ EXIBE AS METRICAS DO DATA_FRAME """
        
        if self.check_df():
            select_df = self.data_frame.select_dtypes(include=['float', 'int'])

            if select_df.empty:
                print("NENHUMA COLUNA COM VALORES INTEIROS")
                return

            for colunn in select_df.columns:
                print("COLUNA: ", colunn)
                print('METRICAS:\n', self.data_frame[colunn].describe())

    def order_df(self):
        """ EXIBE AS METRICAS DO DATA_FRAME """
        
        if self.check_df():
            coluna = input("INFORME A COLUNA: ")
            tp_order = input("INFORME A ORDER DESEJADA:\n1-ASCENDENTE\n2-DESCENDENTE\n: ")
            df_ordenado = self.data_frame.sort_values(coluna, ascending=True if tp_order == '1' else False)
            
            print("DF:\n", df_ordenado)

    def valores_df(self):
        """ EXIBE OS VALORES ABAIXO E ACIMA DA MEDIA DO DATA_FRAME """
        
        if self.check_df():
            df = self.data_frame
            colunas_numericas = df.select_dtypes(include=['float', 'int']).columns

            if colunas_numericas.empty:
                print("NENHUMA COLUNA COM VALORES INTEIROS")
                return

            medias = df[colunas_numericas].mean()

            for coluna in colunas_numericas:
                abaixo_da_media = df[df[coluna] < medias[coluna]][coluna]
                acima_da_media = df[df[coluna] > medias[coluna]][coluna]
                print(f"Coluna {coluna}:\nvalores abaixo da média:\n{abaixo_da_media.unique().tolist()}\nvalores acima da média:\n{acima_da_media.unique().tolist()}")

    def plotar_grafico(self):
        """ CRIA E EXIBE PLOTE DE GRAFICO NO DATA_FRAME """
        
        if self.check_df():
            select_df = self.data_frame.select_dtypes(include=['float', 'int'])

            if select_df.empty:
                print("NENHUMA COLUNA COM VALORES INTEIROS")
                return

            media = select_df.mean()
            self.data_frame[select_df.columns] = select_df.fillna(media)

            df = self.data_frame

            classes = df['class'].unique()

            colunas_aleatorias = select_df.sample(n=2, axis=1).columns

            fig, axs = plt.subplots(nrows=1, ncols=len(classes), figsize=(len(classes)*5, 5))

            for i, c in enumerate(classes):
                axs[i].scatter(df[df['class'] == c][colunas_aleatorias[0]], df[df['class'] == c][colunas_aleatorias[1]])
                axs[i].set_title(c)
                axs[i].set_xlabel(colunas_aleatorias[0])
                axs[i].set_ylabel(colunas_aleatorias[1])

            plt.tight_layout()
            plt.show()


big = BigData()
big.main("1")
