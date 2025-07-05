import unittest
import cv2
import time
import os

from process.funct_models.face_matcher import FaceMatcherModels

def save_summary(test_name: str, summary: dict, path: str):
    with open(f'{path}/summary_{test_name}.txt', 'w') as f:
        f.write(f'Resultados del Test{test_name}\n')
        f.write(f'face matcher correct: {summary["face matcher correct"]}\n')
        f.write(f'face matcher incorrect: {summary["face matcher incorrect"]}\n')
        f.write(f'tiempo de ejecucion: {summary["tiempo total de ejecucion"]} seconds\n')
        f.write(f'Resultado de las imagenes:\n')
        face1_images = [os.path.basename(f) for f in summary['face1 imagen']]
        face2_images = [os.path.basename(f) for f in summary['face2 imagen']]
        for id_image, user_image, coincidence, distance in zip(face1_images, face2_images, summary['coincidence'], summary['distance']):
            f.write(f'{id_image} vs {user_image}: coincidence={coincidence}, distance={distance}\n')
        f.write(f'Mean distance: {sum(summary["distance"]) / len(summary["distance"])}\n')


class TestFaceMatcher(unittest.TestCase):
    def setUp(self):
        self.face_matcher_model = FaceMatcherModels()
    
    def test_face_matcher_face_recognition_model_matcher_images(self): # Funcion para utilizar el modelo face recognition
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_face_recognition_model(face1_image, face2_image) # Hace falta crear las funciones con los modelos

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('face_recognition_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de face recognition model



    def test_face_matcher_vgg_model_matcher_images(self): # Funcion para utilizar el modelo vgg
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_vgg_model(face1_image, face2_image) # Hace falta crear las funciones con los modelos

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('vgg_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de vgg model


    
    def test_face_matcher_facenet_model_matcher_images(self): # Funcion para utilizar el modelo facenet
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_facenet_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('facenet_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de facenet model


    
    def test_face_matcher_facenet512_model_matcher_images(self): # Funcion para utilizar el modelo facenet512
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_facenet512_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('facenet512_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de facenet512 model



    
    def test_face_matcher_openface_model_matcher_images(self): # Funcion para utilizar el modelo openface
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_openface_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('openface_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de openface model



    
    def test_face_matcher_deepface_model_matcher_images(self): # Funcion para utilizar el modelo deepface
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_deepface_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('deepface_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de deepface model



    
    def test_face_matcher_deepid_model_matcher_images(self): # Funcion para utilizar el modelo deepid
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_deepid_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('deepid_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de deepid model



    
    def test_face_matcher_arcface_model_matcher_images(self): # Funcion para utilizar el modelo arcface
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_arcface_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('arcface_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de arcface model



    
    def test_face_matcher_dlib_model_matcher_images(self): # Funcion para utilizar el modelo dlib
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_dlib_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('dlib_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de face dlib model



    
    def test_face_matcher_sface_model_matcher_images(self): # Funcion para utilizar el modelo sface
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_sface_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('sface_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de sface model


            
    def test_face_matcher_ghostfacenet_model_matcher_images(self): # Funcion para utilizar el modelo de ghostfacenet
        face1_input_folder = 'tests/imagenes/similar/face_1/'
        face2_input_folder = 'tests/imagenes/similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_ghostfacenet_model(face1_image, face2_image)

            if coincidence == True:
                resumen_obtenido['face matcher correct'] += 1
            else:
                resumen_obtenido['face matcher incorrect'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)

            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('ghostfacenet_model_matcher', resumen_obtenido, 'tests/resumenes') # Nombre para el resumen de ghostfacenet model



# MODO INVERSO 
#codigo para el testeo de imagenes diferentes


    def test_face_matcher_face_recognition_model_matcher_images_different(self): # Funcion para utilizar el modelo face recognition
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_face_recognition_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1
            
            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('face_recognition_model_matcher_different', resumen_obtenido, 'tests/resumenes') # Nombre del resumen

    def test_face_matcher_vgg_model_matcher_images_different(self): # Funcion para utilizar el modelo vgg
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_vgg_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('vgg_model_matcher_different', resumen_obtenido, 'tests/resumenes')

    def test_face_matcher_facenet_model_matcher_images_different(self): # Funcion para utilizar el modelo facenet
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_facenet_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('facenet_model_matcher_different', resumen_obtenido, 'tests/resumenes')

    def test_face_matcher_facenet512_model_matcher_images_different(self): # Funcion para utilizar el modelo facenet512
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_facenet512_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('facenet512_model_matcher_different', resumen_obtenido, 'tests/resumenes')

    def test_face_matcher_openface_model_matcher_images_different(self): # Funcion para utilizar el modelo openface
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_openface_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('openface_model_matcher_different', resumen_obtenido, 'tests/resumenes')
    
    def test_face_matcher_deepface_model_matcher_images_different(self): # Funcion para utilizar el modelo deepface
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_deepface_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('deepface_model_matcher_different', resumen_obtenido, 'tests/resumenes')

    def test_face_matcher_deepid_model_matcher_images_different(self): # Funcion para utilizar el modelo deepid
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_deepid_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('deepid_model_matcher_different', resumen_obtenido, 'tests/resumenes')
    
    def test_face_matcher_arcface_model_matcher_images_different(self): # Funcion para utilizar el modelo arcface
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_arcface_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('arcface_model_matcher_different', resumen_obtenido, 'tests/resumenes')

    def test_face_matcher_dlib_model_matcher_images_different(self): # Funcion para utilizar el modelo dlib
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_dlib_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('dlib_model_matcher_different', resumen_obtenido, 'tests/resumenes')
    
    def test_face_matcher_sface_model_matcher_images_different(self): # Funcion para utilizar el modelo sface
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_sface_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('sface_model_matcher_different', resumen_obtenido, 'tests/resumenes')
    
    def test_face_matcher_ghostfacenet_model_matcher_images_different(self): # Funcion para utilizar el modelo de ghostfacenet
        face1_input_folder = 'tests/imagenes/no_similar/face_1/'
        face2_input_folder = 'tests/imagenes/no_similar/face_2/'
        inicio_cronometro = time.time()
        resumen_obtenido = {'face matcher correct': 0, 'face matcher incorrect': 0, 'tiempo total de ejecucion': 0,
                            'face1 imagen': [], 'face2 imagen': [], 'coincidence': [], 'distance': []}
        
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        
        for face1_image_path, face2_image_path in zip(face1_images,face2_images ):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)

            coincidence, distance = self.face_matcher_model.face_matching_ghostfacenet_model(face1_image, face2_image)
            if coincidence == True:
                resumen_obtenido['face matcher incorrect'] += 1
            else:
                resumen_obtenido['face matcher correct'] += 1

            resumen_obtenido['face1 imagen'].append(os.path.basename(face1_image_path))
            resumen_obtenido['face2 imagen'].append(os.path.basename(face2_image_path))
            resumen_obtenido['coincidence'].append(coincidence)
            resumen_obtenido['distance'].append(distance)
            fin_cronometro = time.time()
            tiempo_de_ejecucion = fin_cronometro - inicio_cronometro
            resumen_obtenido['tiempo total de ejecucion'] = round(tiempo_de_ejecucion, 3)
            print(resumen_obtenido)
            save_summary('ghostfacenet_model_matcher_different', resumen_obtenido, 'tests/resumenes')
    
    