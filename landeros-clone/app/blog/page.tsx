"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useState } from "react";

const blogPosts = [
  {
    id: 1,
    title: "The Future of AI in Medical Imaging",
    excerpt: "Exploring how artificial intelligence is revolutionizing radiology and diagnostic imaging with unprecedented accuracy and speed.",
    author: "Dr. Sarah Chen",
    date: "Nov 20, 2024",
    category: "AI Technology",
    image: "",
    readTime: "5 min read",
    slug: "future-of-ai-medical-imaging",
  },
  {
    id: 2,
    title: "Understanding Deep Learning in Healthcare",
    excerpt: "A comprehensive guide to how DenseNet121 and other neural networks are transforming medical diagnosis and patient care.",
    author: "Dr. Michael Rodriguez",
    date: "Nov 18, 2024",
    category: "Deep Learning",
    image: "",
    readTime: "8 min read",
    slug: "understanding-deep-learning-healthcare",
  },
  {
    id: 3,
    title: "Pneumonia Detection: AI vs Radiologists",
    excerpt: "Comparing the accuracy and efficiency of AI-powered pneumonia detection against traditional radiologist diagnosis methods.",
    author: "Dr. Emily Watson",
    date: "Nov 15, 2024",
    category: "Case Studies",
    image: "",
    readTime: "6 min read",
    slug: "pneumonia-detection-ai-vs-radiologists",
  },
  {
    id: 4,
    title: "Implementing AI in Clinical Workflows",
    excerpt: "Best practices for integrating AI diagnostic tools into existing hospital and clinic workflows for maximum efficiency.",
    author: "Dr. James Park",
    date: "Nov 12, 2024",
    category: "Implementation",
    image: "",
    readTime: "7 min read",
    slug: "implementing-ai-clinical-workflows",
  },
  {
    id: 5,
    title: "Medical AI Ethics and Patient Privacy",
    excerpt: "Addressing the ethical considerations and privacy concerns in AI-powered medical imaging and diagnosis.",
    author: "Dr. Lisa Anderson",
    date: "Nov 10, 2024",
    category: "Ethics",
    image: "",
    readTime: "10 min read",
    slug: "medical-ai-ethics-privacy",
  },
  {
    id: 6,
    title: "Training Medical AI Models: A Deep Dive",
    excerpt: "Behind the scenes look at how we train our deep learning models on millions of medical images for clinical-grade accuracy.",
    author: "Dr. Robert Kim",
    date: "Nov 8, 2024",
    category: "Technical",
    image: "",
    readTime: "12 min read",
    slug: "training-medical-ai-models",
  },
];

export default function BlogPage() {
  return (
    <main className="min-h-screen relative bg-background-dark">
      <div className="relative z-10">
        <Navbar />
        
        <div className="pt-32 pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="text-center mb-16">
              <h1 className="text-6xl font-display font-bold mb-6 text-foreground">
                Medical AI <span className="text-accent">Insights</span>
              </h1>
              <p className="text-xl text-muted max-w-3xl mx-auto">
                Latest research, case studies, and insights from the world of AI-powered medical imaging
              </p>
            </div>

            {/* Featured Post */}
            <a href={`/blog/${blogPosts[0].slug}`} className="block bg-gradient-to-br from-accent to-accent-dark text-white rounded-3xl p-12 mb-12 cursor-pointer shadow-xl shadow-accent/20 hover:-translate-y-2 transition-transform duration-500">
              <div className="grid md:grid-cols-2 gap-12 items-center">
                <div>
                  <span className="inline-block bg-white/20 px-4 py-2 rounded-full text-sm font-semibold mb-4 backdrop-blur-sm">
                    Featured Article
                  </span>
                  <h2 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
                    {blogPosts[0].title}
                  </h2>
                  <p className="text-white/90 text-lg mb-8 leading-relaxed">
                    {blogPosts[0].excerpt}
                  </p>
                  <div className="flex items-center gap-4 mb-8">
                    <span className="text-sm font-medium">{blogPosts[0].author}</span>
                    <span className="text-white/60">•</span>
                    <span className="text-sm">{blogPosts[0].date}</span>
                    <span className="text-white/60">•</span>
                    <span className="text-sm">{blogPosts[0].readTime}</span>
                  </div>
                  <div className="bg-white text-accent px-8 py-3 rounded-full font-bold hover:bg-white/90 transition-all inline-block shadow-lg">
                    Read Article
                  </div>
                </div>
                <div className="text-9xl text-center transform hover:scale-110 transition-transform duration-500 drop-shadow-2xl">{blogPosts[0].image}</div>
              </div>
            </a>

            {/* Blog Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {blogPosts.slice(1).map((post) => (
                <a key={post.id} href={`/blog/${post.slug}`} className="bg-white rounded-2xl p-8 border border-border card-hover group cursor-pointer shadow-sm hover:shadow-md transition-all">
                  <div className="text-6xl mb-6 transform group-hover:scale-110 transition-transform duration-300">{post.image}</div>
                  
                  <span className="inline-block bg-accent/10 text-accent px-3 py-1 rounded-full text-xs font-bold mb-4 tracking-wide">
                    {post.category}
                  </span>
                  
                  <h3 className="text-2xl font-bold mb-3 text-foreground group-hover:text-accent transition-colors leading-tight">
                    {post.title}
                  </h3>
                  
                  <p className="text-muted mb-6 line-clamp-3 leading-relaxed text-sm">
                    {post.excerpt}
                  </p>
                  
                  <div className="flex items-center justify-between pt-4 border-t border-border mt-auto">
                    <div>
                      <p className="text-sm font-bold text-foreground">{post.author}</p>
                      <p className="text-xs text-muted font-medium">{post.date}</p>
                    </div>
                    <p className="text-xs text-accent font-bold bg-accent/5 px-2 py-1 rounded">{post.readTime}</p>
                  </div>
                </a>
              ))}
            </div>

          </div>
        </div>

        <Footer />
      </div>
    </main>
  );
}
