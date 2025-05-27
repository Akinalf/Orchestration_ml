#!/usr/bin/env python3
"""
Script pour visualiser les résultats du pipeline Kedro
"""
import pickle
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle_file(filepath):
    """Charge un fichier pickle"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def load_json_file(filepath):
    """Charge un fichier JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def display_detection_results():
    """Affiche les résultats de détection YOLO"""
    print("🔍 RÉSULTATS DE DÉTECTION YOLO")
    print("=" * 50)
    
    detection_file = Path("data/07_model_output/detection_results.pkl")
    if detection_file.exists():
        detections = load_pickle_file(detection_file)
        if detections:
            print(f"📊 Nombre total d'images traitées: {len(detections)}")
            
            # Statistiques des détections
            successful_detections = [d for d in detections if d.get('bbox') is not None]
            print(f"✅ Détections réussies: {len(successful_detections)}")
            print(f"❌ Aucune détection: {len(detections) - len(successful_detections)}")
            print(f"📈 Taux de réussite: {len(successful_detections)/len(detections)*100:.1f}%")
            
            if successful_detections:
                confidences = [d['confidence'] for d in successful_detections]
                print(f"🎯 Confiance moyenne: {sum(confidences)/len(confidences):.3f}")
                print(f"📊 Confiance min/max: {min(confidences):.3f} / {max(confidences):.3f}")
                
                # Afficher quelques exemples
                print("\n📝 Exemples de détections:")
                for i, detection in enumerate(successful_detections[:5]):
                    image_name = Path(detection['image_path']).name
                    conf = detection['confidence']
                    bbox = detection['bbox']
                    print(f"  {i+1}. {image_name} - Confiance: {conf:.3f} - BBox: {bbox}")
            
            print("\n" + "=" * 50)
        else:
            print("❌ Impossible de charger les résultats de détection")
    else:
        print("❌ Fichier de détection non trouvé")

def display_ocr_results():
    """Affiche les résultats OCR"""
    print("\n📖 RÉSULTATS OCR")
    print("=" * 50)
    
    ocr_file = Path("data/07_model_output/ocr_results.json")
    if ocr_file.exists():
        ocr_results = load_json_file(ocr_file)
        if ocr_results:
            print(f"📊 Nombre total de ROI traitées: {len(ocr_results)}")
            
            # Textes extraits avec succès
            successful_ocr = [r for r in ocr_results if r.get('detected_text', '').strip()]
            print(f"✅ Textes extraits: {len(successful_ocr)}")
            print(f"❌ Échecs OCR: {len(ocr_results) - len(successful_ocr)}")
            
            if successful_ocr:
                confidences = [r['confidence'] for r in successful_ocr if 'confidence' in r]
                if confidences:
                    print(f"🎯 Confiance OCR moyenne: {sum(confidences)/len(confidences):.1f}%")
                
                # Textes les plus fréquents
                texts = [r['detected_text'] for r in successful_ocr]
                text_counts = {}
                for text in texts:
                    text_counts[text] = text_counts.get(text, 0) + 1
                
                print("\n📝 Textes détectés les plus fréquents:")
                sorted_texts = sorted(text_counts.items(), key=lambda x: x[1], reverse=True)
                for text, count in sorted_texts[:10]:
                    print(f"  '{text}' : {count} fois")
                
                print("\n📋 Exemples de textes extraits:")
                for i, result in enumerate(successful_ocr[:10]):
                    image_name = Path(result['image_path']).name
                    text = result['detected_text']
                    conf = result.get('confidence', 0)
                    print(f"  {i+1}. {image_name} → '{text}' (confiance: {conf:.1f}%)")
            
            print("\n" + "=" * 50)
        else:
            print("❌ Impossible de charger les résultats OCR")
    else:
        print("❌ Fichier OCR non trouvé")

def display_final_report():
    """Affiche le rapport final"""
    print("\n📈 RAPPORT FINAL")
    print("=" * 50)
    
    report_file = Path("data/08_reporting/final_report.json")
    if report_file.exists():
        report = load_json_file(report_file)
        if report:
            # Résumé du pipeline
            if 'pipeline_summary' in report:
                summary = report['pipeline_summary']
                print("🚀 RÉSUMÉ DU PIPELINE:")
                print(f"  📊 Images traitées: {summary.get('total_images_processed', 'N/A')}")
                print(f"  🔍 Taux détection: {summary.get('detection_success_rate', 0)*100:.1f}%")
                print(f"  📖 Taux OCR: {summary.get('ocr_success_rate', 0)*100:.1f}%")
                print(f"  🎯 Succès end-to-end: {summary.get('end_to_end_success_rate', 0)*100:.1f}%")
            
            # Configuration utilisée
            if 'configuration_used' in report:
                config = report['configuration_used']
                print(f"\n⚙️ CONFIGURATION:")
                print(f"  🎯 Seuil confiance YOLO: {config.get('yolo_confidence_threshold', 'N/A')}")
                print(f"  📖 Langue OCR: {config.get('ocr_language', 'N/A')}")
                print(f"  🔍 Confiance OCR min: {config.get('ocr_min_confidence', 'N/A')}")
            
            # Recommandations
            if 'recommendations' in report and report['recommendations']:
                print(f"\n💡 RECOMMANDATIONS:")
                for i, rec in enumerate(report['recommendations'], 1):
                    print(f"  {i}. {rec}")
            
            print("\n" + "=" * 50)
        else:
            print("❌ Impossible de charger le rapport final")
    else:
        print("❌ Rapport final non trouvé")

def display_evaluation_metrics():
    """Affiche les métriques d'évaluation"""
    print("\n📊 MÉTRIQUES D'ÉVALUATION")
    print("=" * 50)
    
    metrics_file = Path("data/08_reporting/evaluation_metrics.json")
    if metrics_file.exists():
        metrics = load_json_file(metrics_file)
        if metrics:
            print("🎯 MÉTRIQUES DISPONIBLES:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_rate') or key.endswith('_ratio'):
                        print(f"  {key}: {value*100:.1f}%")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
            print("\n" + "=" * 50)
        else:
            print("❌ Impossible de charger les métriques")
    else:
        print("❌ Métriques non trouvées")

def create_summary_visualization():
    """Crée une visualisation des résultats"""
    try:
        # Charger les données
        detection_file = Path("data/07_model_output/detection_results.pkl")
        ocr_file = Path("data/07_model_output/ocr_results.json")
        
        if detection_file.exists() and ocr_file.exists():
            detections = load_pickle_file(detection_file)
            ocr_results = load_json_file(ocr_file)
            
            if detections and ocr_results:
                # Créer un graphique simple
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Graphique 1: Répartition des détections
                successful_det = len([d for d in detections if d.get('bbox') is not None])
                failed_det = len(detections) - successful_det
                
                ax1.pie([successful_det, failed_det], 
                       labels=['Détections réussies', 'Échecs'], 
                       autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'])
                ax1.set_title('Répartition des détections YOLO')
                
                # Graphique 2: Répartition OCR
                successful_ocr = len([r for r in ocr_results if r.get('detected_text', '').strip()])
                failed_ocr = len(ocr_results) - successful_ocr
                
                ax2.pie([successful_ocr, failed_ocr], 
                       labels=['OCR réussi', 'Échecs'], 
                       autopct='%1.1f%%',
                       colors=['#3498db', '#f39c12'])
                ax2.set_title('Répartition des résultats OCR')
                
                plt.tight_layout()
                plt.savefig('data/08_reporting/results_summary.png', dpi=300, bbox_inches='tight')
                print("📊 Graphique sauvegardé: data/08_reporting/results_summary.png")
                plt.show()
                
    except Exception as e:
        print(f"❌ Erreur lors de la création du graphique: {e}")

def main():
    """Fonction principale"""
    print("🔍 VISUALISATION DES RÉSULTATS DU PIPELINE KEDRO")
    print("=" * 60)
    
    # Vérifier que les dossiers existent
    output_dir = Path("data/07_model_output")
    reporting_dir = Path("data/08_reporting")
    
    if not output_dir.exists():
        print("❌ Dossier de sortie non trouvé. Avez-vous lancé le pipeline ?")
        return
    
    # Afficher tous les résultats
    display_detection_results()
    display_ocr_results()
    display_evaluation_metrics()
    display_final_report()
    
    # Créer une visualisation
    print("\n📊 Création d'une visualisation...")
    create_summary_visualization()
    
    print("\n✅ Analyse terminée !")
    print("💡 Consultez aussi le fichier: data/08_reporting/results_summary.png")

if __name__ == "__main__":
    main()