'''
Visualization for Anomaly Detection Results
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path


class AnomalyVisualizer:
    '''Create visualizations for anomaly detection results'''
    
    def __init__(self, output_dir: str = 'assets'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_time_series_with_anomalies(self, 
                                       data: np.ndarray,
                                       anomalies: List[Dict],
                                       title: str = 'Anomaly Detection Results',
                                       save: bool = True) -> str:
        '''Plot time series with detected anomalies highlighted'''
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot normal data
        ax.plot(data, linewidth=1.5, alpha=0.7, label='Time Series', color='steelblue')
        
        # Highlight anomalies
        anomaly_indices = [a['index'] for a in anomalies]
        anomaly_values = [data[i] for i in anomaly_indices]
        
        ax.scatter(anomaly_indices, anomaly_values, 
                  color='red', s=100, marker='x', linewidths=3,
                  label=f'Anomalies ({len(anomalies)})', zorder=5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save:
            filepath = self.output_dir / 'anomaly_detection.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def plot_anomaly_scores(self,
                           scores: np.ndarray,
                           threshold: float = 0.5,
                           title: str = 'Anomaly Scores Over Time',
                           save: bool = True) -> str:
        '''Plot anomaly scores with threshold'''
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot scores
        ax.plot(scores, linewidth=1.5, alpha=0.8, color='green', label='Anomaly Score')
        
        # Threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Threshold ({threshold})')
        
        # Fill area above threshold
        ax.fill_between(range(len(scores)), threshold, scores,
                       where=(scores > threshold), alpha=0.3, color='red',
                       label='Anomaly Region')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save:
            filepath = self.output_dir / 'anomaly_scores.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def plot_detector_comparison(self,
                                results_dict: Dict[str, List[Dict]],
                                data: np.ndarray,
                                save: bool = True) -> str:
        '''Compare multiple detectors side by side'''
        n_detectors = len(results_dict)
        fig, axes = plt.subplots(n_detectors, 1, figsize=(14, 4*n_detectors))
        
        if n_detectors == 1:
            axes = [axes]
            
        for ax, (detector_name, anomalies) in zip(axes, results_dict.items()):
            # Plot data
            ax.plot(data, linewidth=1, alpha=0.7, color='steelblue')
            
            # Plot anomalies
            anomaly_indices = [a['index'] for a in anomalies]
            anomaly_values = [data[i] for i in anomaly_indices]
            
            ax.scatter(anomaly_indices, anomaly_values,
                      color='red', s=80, marker='x', linewidths=2)
            
            ax.set_title(f'{detector_name}: {len(anomalies)} anomalies detected',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.3)
            
        axes[-1].set_xlabel('Time Index')
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'detector_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def plot_distribution_analysis(self,
                                   normal_data: np.ndarray,
                                   anomalies: np.ndarray,
                                   save: bool = True) -> str:
        '''Plot distribution of normal vs anomalous data'''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram comparison
        ax1.hist(normal_data, bins=50, alpha=0.7, color='blue', label='Normal', density=True)
        ax1.hist(anomalies, bins=30, alpha=0.7, color='red', label='Anomalies', density=True)
        ax1.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot
        data_to_plot = [normal_data, anomalies]
        labels = ['Normal', 'Anomalies']
        ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'distribution_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def create_dashboard(self,
                        data: np.ndarray,
                        anomalies: List[Dict],
                        scores: np.ndarray,
                        metrics: Dict,
                        save: bool = True) -> str:
        '''Create comprehensive dashboard'''
        fig = plt.figure(figsize=(16, 10))
        
        # Layout: 2x2 grid
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        
        # 1. Time series with anomalies
        ax1.plot(data, linewidth=1.5, alpha=0.7, color='steelblue')
        anomaly_indices = [a['index'] for a in anomalies]
        anomaly_values = [data[i] for i in anomaly_indices]
        ax1.scatter(anomaly_indices, anomaly_values, 
                   color='red', s=80, marker='x', linewidths=2)
        ax1.set_title('Time Series with Anomalies', fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(alpha=0.3)
        
        # 2. Anomaly scores
        ax2.plot(scores, linewidth=1.5, color='green', alpha=0.8)
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2)
        ax2.fill_between(range(len(scores)), 0.5, scores,
                        where=(scores > 0.5), alpha=0.3, color='red')
        ax2.set_title('Anomaly Scores', fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)
        
        # 3. Distribution
        normal_mask = np.ones(len(data), dtype=bool)
        normal_mask[anomaly_indices] = False
        normal_data = data[normal_mask]
        anomaly_data = data[anomaly_indices]
        
        ax3.hist(normal_data, bins=50, alpha=0.7, color='blue', label='Normal', density=True)
        ax3.hist(anomaly_data, bins=20, alpha=0.7, color='red', label='Anomalies', density=True)
        ax3.set_title('Value Distribution', fontweight='bold')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Performance metrics
        ax4.axis('off')
        metrics_text = [
            'Performance Metrics',
            '=' * 30,
            f'Precision: {metrics.get("precision", 0):.3f}',
            f'Recall: {metrics.get("recall", 0):.3f}',
            f'F1 Score: {metrics.get("f1_score", 0):.3f}',
            '',
            'Detection Summary',
            '=' * 30,
            f'Total Points: {len(data)}',
            f'Anomalies: {len(anomalies)} ({len(anomalies)/len(data)*100:.1f}%)',
            f'True Positives: {metrics.get("true_positives", "N/A")}',
            f'False Positives: {metrics.get("false_positives", "N/A")}',
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(metrics_text),
                verticalalignment='top',
                fontfamily='monospace',
                fontsize=11)
        
        plt.suptitle('Anomaly Detection Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'dashboard.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None


def demo():
    '''Demo visualizations'''
    from detectors.pipeline import generate_synthetic_data
    
    print('Anomaly Visualization Demo')
    print('=' * 50)
    
    # Generate data
    data, true_labels = generate_synthetic_data(n_points=500, anomaly_rate=0.05)
    
    # Mock anomalies
    anomaly_indices = np.where(true_labels)[0]
    anomalies = [{'index': int(i), 'score': 0.8} for i in anomaly_indices]
    
    # Mock scores
    scores = np.random.random(len(data)) * 0.4
    scores[anomaly_indices] = np.random.random(len(anomaly_indices)) * 0.4 + 0.6
    
    # Create visualizer
    viz = AnomalyVisualizer()
    
    print('\nGenerating visualizations...')
    
    viz.plot_time_series_with_anomalies(data, anomalies)
    print('  ✅ Time series plot')
    
    viz.plot_anomaly_scores(scores)
    print('  ✅ Anomaly scores plot')
    
    normal_data = data[~true_labels]
    anomaly_data = data[true_labels]
    viz.plot_distribution_analysis(normal_data, anomaly_data)
    print('  ✅ Distribution analysis')
    
    metrics = {
        'precision': 0.85,
        'recall': 0.78,
        'f1_score': 0.81,
        'true_positives': 19,
        'false_positives': 3
    }
    viz.create_dashboard(data, anomalies, scores, metrics)
    print('  ✅ Dashboard')
    
    print('\n✅ All visualizations saved to assets/')


if __name__ == '__main__':
    demo()
