"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Contact from "@/components/Community";

export default function ContactPage() {
  return (
    <main className="min-h-screen relative">
      <div className="relative z-10">
        <Navbar />
        <div className="pt-32">
          <Contact />
        </div>
        <Footer />
      </div>
    </main>
  );
}
