"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Link from "next/link";

export default function BlogPost() {
  return (
    <main className="min-h-screen relative bg-background-dark flex flex-col">
      <Navbar />
      
      <div className="flex-grow flex items-center justify-center px-6 pt-32 pb-20">
        <div className="max-w-3xl w-full text-center">
          <div className="mb-8 relative inline-block">
            <div className="absolute -inset-4 bg-accent/20 rounded-full blur-xl animate-pulse"></div>
            <div className="relative bg-white p-6 rounded-full shadow-xl">
              <span className="text-6xl">✍️</span>
            </div>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold mb-6 text-foreground">
            Article Coming <span className="text-accent">Soon</span>
          </h1>
          
          <p className="text-xl text-muted mb-12 leading-relaxed">
            Our medical experts are currently reviewing and finalizing this research paper. 
            Please check back later for the full analysis.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link 
              href="/blog" 
              className="bg-accent text-white px-8 py-4 rounded-full font-bold hover:bg-accent-dark transition-all hover:-translate-y-1 shadow-lg shadow-accent/20 w-full sm:w-auto"
            >
              Back to Blog
            </Link>
            <Link 
              href="/" 
              className="bg-white text-foreground border border-border px-8 py-4 rounded-full font-bold hover:border-accent hover:text-accent transition-all w-full sm:w-auto"
            >
              Return Home
            </Link>
          </div>
        </div>
      </div>

      <Footer />
    </main>
  );
}
