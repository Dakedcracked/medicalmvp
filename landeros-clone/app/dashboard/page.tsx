"use client";

import { useEffect, useState } from "react";
import { AuthService } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

interface PredictionHistory {
  id: number;
  prediction: string;
  confidence: number;
  created_at: string;
}

export default function DashboardPage() {
  const [user, setUser] = useState<any>(null);
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalScans: 0,
    avgConfidence: 0,
    criticalFindings: 0,
  });
  const router = useRouter();

  useEffect(() => {
    const checkAuth = async () => {
      const authenticated = AuthService.isAuthenticated();
      if (!authenticated) {
        router.push("/");
        return;
      }

      const currentUser = AuthService.getUser();
      setUser(currentUser);

      // Fetch prediction history
      try {
        const token = AuthService.getToken();
        const response = await fetch(`${API_URL}/api/predictions/history`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setHistory(data.history || []);
          
          // Calculate stats
          const total = data.history.length;
          const avgConf = data.history.reduce((sum: number, p: any) => sum + p.confidence, 0) / total || 0;
          const critical = data.history.filter((p: any) => p.confidence > 0.9).length;
          
          setStats({
            totalScans: total,
            avgConfidence: avgConf,
            criticalFindings: critical,
          });
        }
      } catch (error) {
        console.error("Failed to fetch history:", error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, [router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-muted">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen relative">
      <div className="relative z-10">
        <Navbar />
        
        <div className="pt-32 pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            {/* Welcome Section */}
            <div className="mb-12">
              <h1 className="text-5xl font-bold mb-4">
                Welcome back, <span className="text-accent">{user?.name}</span>
              </h1>
              <p className="text-muted text-lg">
                Your medical AI analysis dashboard
              </p>
            </div>

            {/* Stats Grid */}
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="bg-white rounded-2xl p-8 border border-border card-hover">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center">
                    <svg className="w-6 h-6 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                </div>
                <p className="text-3xl font-bold mb-2">{stats.totalScans}</p>
                <p className="text-muted">Total Scans</p>
              </div>

              <div className="bg-white rounded-2xl p-8 border border-border card-hover">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-green-500/10 rounded-xl flex items-center justify-center">
                    <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                </div>
                <p className="text-3xl font-bold mb-2">{(stats.avgConfidence * 100).toFixed(1)}%</p>
                <p className="text-muted">Avg Confidence</p>
              </div>

              <div className="bg-white rounded-2xl p-8 border border-border card-hover">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-red-500/10 rounded-xl flex items-center justify-center">
                    <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                </div>
                <p className="text-3xl font-bold mb-2">{stats.criticalFindings}</p>
                <p className="text-muted">Critical Findings</p>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <a href="/analyze" className="bg-gradient-to-br from-accent to-accent-dark text-white rounded-2xl p-8 card-hover group">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold">New Analysis</h3>
                  <svg className="w-8 h-8 group-hover:translate-x-2 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
                <p className="text-white/80">Upload and analyze a new medical image</p>
              </a>

              <a href="/dashboard/profile" className="bg-white rounded-2xl p-8 border border-border card-hover group">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold">Profile</h3>
                  <svg className="w-8 h-8 text-accent group-hover:translate-x-2 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <p className="text-muted">Manage your account and preferences</p>
              </a>

              <a href="/blog" className="bg-white rounded-2xl p-8 border border-border card-hover group">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold">Research & Blog</h3>
                  <svg className="w-8 h-8 text-accent group-hover:translate-x-2 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <p className="text-muted">Read detailed research papers and articles</p>
              </a>
            </div>

            {/* Recent Scans */}
            <div className="bg-white rounded-2xl p-8 border border-border">
              <h2 className="text-2xl font-bold mb-6">Recent Scans</h2>
              
              {history.length === 0 ? (
                <div className="text-center py-12">
                  <svg className="w-16 h-16 text-muted mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-muted">No scans yet. Upload your first medical image!</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {history.slice(0, 10).map((scan) => (
                    <div key={scan.id} className="flex items-center justify-between p-4 bg-background-dark rounded-xl hover:bg-accent/5 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                          scan.confidence > 0.9 ? 'bg-red-500/10' : 
                          scan.confidence > 0.7 ? 'bg-yellow-500/10' : 
                          'bg-green-500/10'
                        }`}>
                          <svg className={`w-6 h-6 ${
                            scan.confidence > 0.9 ? 'text-red-500' : 
                            scan.confidence > 0.7 ? 'text-yellow-500' : 
                            'text-green-500'
                          }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <div>
                          <p className="font-semibold">{scan.prediction}</p>
                          <p className="text-sm text-muted">{new Date(scan.created_at).toLocaleDateString()}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-accent">{(scan.confidence * 100).toFixed(1)}%</p>
                        <p className="text-xs text-muted">Confidence</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </main>
  );
}
