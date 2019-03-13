# StressClassifier

« Classification 2-classes en utilisant SVM  » 

%By Alaeddine BOUJNAH

Dans ce test nous avons abordé un problème de classification 2-classes puisque la base de données s’agit de plusieurs échantillons mésurés (1420 samples) que bien évidemment appartiennent à un nombre de classe égale à deux.  Afin d’effectuer cette classification on a eu recours à un algorithme supervisé de Machine Learning SVM de classification parceque c’est une simple méthode que peut nous fournir un bon clustering des données et quand il s'agit un dual probléme,l'usage de kernel trick rend la classification plus efficace et l'optimal margin gap entre les hyperplans séparés elle fait une meilleure prédiction avec un test data ce que le rend plus robuste et avec une haute accurancy.
Pour bien préparer les données au clustering,on sélectionne des attributs les plus significatifs:le choix des attributs est très important, pour ne pas tomber dans la redondance et pour avoir une meilleur classification. Dans notre cas admettons que nos 4 features sont significatifs indépendants (bpm,rmssd,bsv,sdnn).

#Pyhton 
using scikit-learn de Python
Préparer train/test data,créer une instance de SVM et fit out data avec un linear kernel et un paramétre de régularisation=0.1
Appliquer la prédiction et calculer la matrice de confusion 
On obtient un accurancy score de 94%
