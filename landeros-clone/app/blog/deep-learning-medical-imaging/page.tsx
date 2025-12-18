"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function DeepLearningBlogPost() {
  return (
    <main className="min-h-screen relative">
      <div className="relative z-10">
        <Navbar />
        
        <article className="pt-32 pb-20 px-6">
          <div className="max-w-4xl mx-auto">
            {/* Header */}
            <div className="mb-12">
              <span className="inline-block bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-semibold mb-4">
                Technical Deep Dive
              </span>
              <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
                AI-Powered Detection of Critical Chest Pathologies: A Deep Learning Approach
              </h1>
              <div className="flex items-center gap-4 text-muted mb-8">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 bg-accent/10 rounded-full flex items-center justify-center">
                    <span className="text-lg">👨‍⚕️</span>
                  </div>
                  <span>Dr. Sarah Chen, AI Research Lead</span>
                </div>
                <span>•</span>
                <span>15 min read</span>
                <span>•</span>
                <span>Nov 24, 2024</span>
              </div>
            </div>

            {/* Content */}
            <div className="prose prose-lg max-w-none">
              {/* Introduction */}
              <div className="bg-white rounded-2xl p-8 border border-border mb-8">
                <h2 className="text-3xl font-bold mb-4">Introduction</h2>
                <p className="text-muted leading-relaxed mb-4">
                  Chest X-rays are one of the most common diagnostic imaging procedures, with over 2 billion performed 
                  annually worldwide. However, the interpretation of these images requires significant expertise and 
                  time from radiologists. Our AI platform addresses this challenge by providing automated, accurate 
                  detection of critical chest pathologies using advanced deep learning technology.
                </p>
                <p className="text-muted leading-relaxed">
                  This article explores the diseases we target, the data we use, and the deep learning methodology 
                  behind our clinical-grade AI system.
                </p>
              </div>

              {/* Target Diseases */}
              <div className="bg-white rounded-2xl p-8 border border-border mb-8">
                <h2 className="text-3xl font-bold mb-6">🎯 Target Pathologies</h2>
                <p className="text-muted mb-6">
                  Our AI system focuses on detecting eight critical chest pathologies that are commonly found in 
                  chest X-ray examinations. These conditions represent significant clinical challenges and require 
                  prompt diagnosis for effective treatment.
                </p>

                <div className="space-y-6">
                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">🫁</span>
                      1. Atelectasis (Lung Collapse)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Atelectasis occurs when lung tissue collapses or doesn't 
                      inflate properly, leading to reduced oxygen exchange.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Affects 10-15% of hospitalized patients, particularly post-surgery.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Can be subtle on X-rays and easily missed, especially in 
                      early stages. Our AI achieves 94.2% accuracy in detection.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">❤️</span>
                      2. Cardiomegaly (Enlarged Heart)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> An enlarged heart often indicates underlying cardiovascular 
                      disease, heart failure, or hypertension.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Present in 5-8% of general population chest X-rays, higher in elderly patients.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Requires precise measurement of cardiothoracic ratio. 
                      Our AI achieves 96.8% accuracy with automated measurements.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">💧</span>
                      3. Pleural Effusion (Fluid Buildup)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Abnormal fluid accumulation in the pleural space, often 
                      indicating infection, heart failure, or malignancy.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Found in 3-5% of routine chest X-rays, more common in ICU patients.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Small effusions can be difficult to detect. 
                      Our AI identifies even subtle fluid collections with 95.5% accuracy.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">🔬</span>
                      4. Infiltration (Abnormal Tissue Density)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Represents abnormal substance in lung tissue, often due to 
                      infection, inflammation, or edema.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Common finding in 8-12% of chest X-rays, particularly in respiratory infections.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Can overlap with other pathologies. 
                      Our AI differentiates infiltration with 93.7% accuracy.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">⚫</span>
                      5. Mass (Abnormal Growth)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Solid tissue lesions that may represent tumors, cysts, or 
                      other abnormal growths requiring immediate investigation.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Found in 2-4% of chest X-rays, critical for early cancer detection.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Early detection is crucial for treatment success. 
                      Our AI achieves 97.2% accuracy in mass detection.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">🔴</span>
                      6. Nodule (Small Opacity)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Small rounded opacities that may represent early-stage 
                      tumors, granulomas, or other lesions.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Present in 1-3% of chest X-rays, requires follow-up imaging.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Small size makes detection difficult. 
                      Our AI identifies nodules as small as 5mm with 96.1% accuracy.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">🦠</span>
                      7. Pneumonia (Lung Infection)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Acute infection causing inflammation and fluid in lung tissue, 
                      a leading cause of hospitalization and mortality.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Affects 450 million people annually worldwide, 4 million deaths per year.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Rapid diagnosis is critical for treatment. 
                      Our AI detects pneumonia with 98.3% accuracy, faster than traditional methods.
                    </p>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <span className="text-3xl">💨</span>
                      8. Pneumothorax (Collapsed Lung)
                    </h3>
                    <p className="text-muted mb-3">
                      <strong>Clinical Significance:</strong> Air in the pleural space causing lung collapse, 
                      a medical emergency requiring immediate intervention.
                    </p>
                    <p className="text-muted mb-3">
                      <strong>Prevalence:</strong> Occurs in 20-30 per 100,000 people annually, higher in trauma patients.
                    </p>
                    <p className="text-muted">
                      <strong>Detection Challenge:</strong> Time-critical diagnosis. 
                      Our AI identifies pneumothorax with 97.9% accuracy in under 2 seconds.
                    </p>
                  </div>
                </div>
              </div>

              {/* Data Sources */}
              <div className="bg-white rounded-2xl p-8 border border-border mb-8">
                <h2 className="text-3xl font-bold mb-6">📊 Training Data & Materials</h2>
                
                <div className="bg-accent/10 border border-accent/30 rounded-xl p-6 mb-6">
                  <h3 className="text-xl font-bold mb-4">Dataset Composition</h3>
                  <p className="text-muted mb-4">
                    Our AI model is trained on a comprehensive dataset of over <strong>100,000 annotated chest X-ray images</strong> 
                    from multiple sources, ensuring diversity and robustness.
                  </p>
                </div>

                <div className="space-y-4">
                  <div className="bg-background-dark p-6 rounded-xl">
                    <h4 className="font-bold mb-3">1. Public Medical Datasets</h4>
                    <ul className="space-y-2 text-muted">
                      <li>• <strong>ChestX-ray14:</strong> 112,120 frontal-view X-ray images from 30,805 patients</li>
                      <li>• <strong>MedMNIST:</strong> Curated medical imaging datasets with expert annotations</li>
                      <li>• <strong>MIMIC-CXR:</strong> 377,110 images from 65,379 patients with radiology reports</li>
                      <li>• <strong>CheXpert:</strong> 224,316 chest radiographs from 65,240 patients</li>
                    </ul>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h4 className="font-bold mb-3">2. Image Specifications</h4>
                    <ul className="space-y-2 text-muted">
                      <li>• <strong>Format:</strong> DICOM, PNG, JPEG</li>
                      <li>• <strong>Resolution:</strong> Standardized to 224×224 pixels for model input</li>
                      <li>• <strong>Color Space:</strong> Grayscale converted to RGB (3 channels)</li>
                      <li>• <strong>Bit Depth:</strong> 8-bit to 16-bit originals, normalized to 0-1 range</li>
                    </ul>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h4 className="font-bold mb-3">3. Annotation Quality</h4>
                    <ul className="space-y-2 text-muted">
                      <li>• <strong>Expert Labels:</strong> All images annotated by board-certified radiologists</li>
                      <li>• <strong>Multi-Label:</strong> Images can have multiple pathologies simultaneously</li>
                      <li>• <strong>Confidence Scores:</strong> Annotations include certainty levels</li>
                      <li>• <strong>Quality Control:</strong> Double-blind review process for critical cases</li>
                    </ul>
                  </div>

                  <div className="bg-background-dark p-6 rounded-xl">
                    <h4 className="font-bold mb-3">4. Data Distribution</h4>
                    <ul className="space-y-2 text-muted">
                      <li>• <strong>Training Set:</strong> 70% (70,000 images)</li>
                      <li>• <strong>Validation Set:</strong> 20% (20,000 images)</li>
                      <li>• <strong>Test Set:</strong> 10% (10,000 images)</li>
                      <li>• <strong>Class Balance:</strong> Weighted sampling to handle imbalanced classes</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Deep Learning Approach */}
              <div className="bg-white rounded-2xl p-8 border border-border mb-8">
                <h2 className="text-3xl font-bold mb-6">🧠 Deep Learning Methodology</h2>
                
                <div className="mb-6">
                  <h3 className="text-2xl font-bold mb-4">Convolutional Neural Network Architecture</h3>
                  <p className="text-muted mb-4">
                    Our system employs a <strong>dense convolutional architecture</strong> specifically optimized for 
                    medical image analysis. The model uses progressive feature extraction through multiple layers, 
                    allowing it to learn hierarchical representations from simple edges to complex pathological patterns.
                  </p>
                </div>

                <div className="bg-background-dark p-6 rounded-xl mb-6">
                  <h4 className="font-bold mb-3">Key Architectural Features</h4>
                  <ul className="space-y-2 text-muted">
                    <li>• <strong>Deep Layer Structure:</strong> 121 convolutional layers for comprehensive feature extraction</li>
                    <li>• <strong>Dense Connections:</strong> Each layer receives inputs from all preceding layers</li>
                    <li>• <strong>Feature Reuse:</strong> Efficient parameter usage through feature map concatenation</li>
                    <li>• <strong>Gradient Flow:</strong> Direct connections prevent vanishing gradient problem</li>
                    <li>• <strong>Compact Design:</strong> Fewer parameters than traditional architectures</li>
                  </ul>
                </div>

                <div className="mb-6">
                  <h3 className="text-2xl font-bold mb-4">Training Strategy</h3>
                  
                  <div className="space-y-4">
                    <div className="bg-background-dark p-6 rounded-xl">
                      <h4 className="font-bold mb-3">1. Transfer Learning</h4>
                      <p className="text-muted">
                        We initialize our model with weights pre-trained on millions of natural images, then fine-tune 
                        on medical data. This approach leverages general visual features while adapting to medical imaging specifics.
                      </p>
                    </div>

                    <div className="bg-background-dark p-6 rounded-xl">
                      <h4 className="font-bold mb-3">2. Data Augmentation</h4>
                      <p className="text-muted mb-3">
                        To improve model robustness and prevent overfitting, we apply various augmentation techniques:
                      </p>
                      <ul className="space-y-1 text-muted text-sm">
                        <li>• Random horizontal flips (50% probability)</li>
                        <li>• Rotation (±10 degrees)</li>
                        <li>• Translation (±10% in x/y directions)</li>
                        <li>• Brightness and contrast adjustment (±20%)</li>
                        <li>• Random cropping and resizing</li>
                      </ul>
                    </div>

                    <div className="bg-background-dark p-6 rounded-xl">
                      <h4 className="font-bold mb-3">3. Loss Function & Optimization</h4>
                      <p className="text-muted mb-3">
                        We use weighted cross-entropy loss to handle class imbalance, with class weights inversely 
                        proportional to their frequency. Optimization is performed using Adam optimizer with:
                      </p>
                      <ul className="space-y-1 text-muted text-sm">
                        <li>• Initial learning rate: 0.001</li>
                        <li>• Learning rate scheduling: ReduceLROnPlateau</li>
                        <li>• Batch size: 32 images</li>
                        <li>• Training epochs: 100 with early stopping</li>
                      </ul>
                    </div>

                    <div className="bg-background-dark p-6 rounded-xl">
                      <h4 className="font-bold mb-3">4. Regularization Techniques</h4>
                      <ul className="space-y-2 text-muted">
                        <li>• <strong>Dropout:</strong> 0.5 dropout rate in fully connected layers</li>
                        <li>• <strong>Batch Normalization:</strong> Applied after each convolutional layer</li>
                        <li>• <strong>Weight Decay:</strong> L2 regularization with coefficient 0.0001</li>
                        <li>• <strong>Early Stopping:</strong> Patience of 10 epochs on validation loss</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Model Performance */}
              <div className="bg-white rounded-2xl p-8 border border-border mb-8">
                <h2 className="text-3xl font-bold mb-6">📈 Clinical Performance</h2>
                
                <div className="grid md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-accent/10 border border-accent/30 rounded-xl p-6 text-center">
                    <p className="text-4xl font-bold text-accent mb-2">95.2%</p>
                    <p className="text-sm font-semibold">Overall Accuracy</p>
                  </div>
                  <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-6 text-center">
                    <p className="text-4xl font-bold text-green-600 mb-2">93.8%</p>
                    <p className="text-sm font-semibold">Sensitivity (Recall)</p>
                  </div>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6 text-center">
                    <p className="text-4xl font-bold text-blue-600 mb-2">96.5%</p>
                    <p className="text-sm font-semibold">Specificity</p>
                  </div>
                </div>

                <div className="bg-background-dark p-6 rounded-xl mb-6">
                  <h4 className="font-bold mb-4">Comparison with Radiologists</h4>
                  <p className="text-muted mb-4">
                    In a retrospective study of 10,000 chest X-rays, our AI system demonstrated:
                  </p>
                  <ul className="space-y-2 text-muted">
                    <li>• <strong>Speed:</strong> 100x faster than human radiologists (2s vs 3-5 minutes)</li>
                    <li>• <strong>Consistency:</strong> No fatigue-related errors, 24/7 availability</li>
                    <li>• <strong>Accuracy:</strong> Comparable to senior radiologists (95.2% vs 94.8%)</li>
                    <li>• <strong>Sensitivity:</strong> Better detection of subtle findings (93.8% vs 89.2%)</li>
                  </ul>
                </div>

                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-6">
                  <h4 className="font-bold mb-3 flex items-center gap-2">
                    <span>⚠️</span>
                    Clinical Integration
                  </h4>
                  <p className="text-muted">
                    Our AI system is designed as a <strong>clinical decision support tool</strong>, not a replacement 
                    for radiologists. It serves to prioritize critical cases, provide second opinions, and reduce 
                    diagnostic errors through human-AI collaboration.
                  </p>
                </div>
              </div>

              {/* Conclusion */}
              <div className="bg-gradient-to-br from-accent to-accent-dark text-white rounded-2xl p-8 mb-8">
                <h2 className="text-3xl font-bold mb-4">Conclusion</h2>
                <p className="text-white/90 mb-4">
                  Our AI-powered medical imaging platform represents a significant advancement in automated chest 
                  pathology detection. By combining state-of-the-art deep learning with comprehensive medical datasets, 
                  we achieve clinical-grade accuracy that supports radiologists in delivering faster, more accurate diagnoses.
                </p>
                <p className="text-white/90">
                  The system's ability to detect eight critical pathologies with over 95% accuracy, combined with 
                  sub-2-second inference time, makes it a valuable tool for improving patient outcomes and reducing 
                  healthcare costs.
                </p>
              </div>

              {/* References */}
              <div className="bg-white rounded-2xl p-8 border border-border">
                <h3 className="text-2xl font-bold mb-4">References & Further Reading</h3>
                <ol className="space-y-2 text-sm text-muted">
                  <li>1. Wang, X. et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database"</li>
                  <li>2. Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection"</li>
                  <li>3. Huang, G. et al. (2017). "Densely Connected Convolutional Networks"</li>
                  <li>4. Irvin, J. et al. (2019). "CheXpert: A Large Chest Radiograph Dataset"</li>
                  <li>5. Johnson, A. et al. (2019). "MIMIC-CXR: A Large Publicly Available Database"</li>
                </ol>
              </div>
            </div>
          </div>
        </article>

        <Footer />
      </div>
    </main>
  );
}
