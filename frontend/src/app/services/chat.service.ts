import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private apiUrl = 'http://localhost:8000/chat';
  private sessionId = this.generateSessionId();

  constructor(private http: HttpClient) {}

  sendQuestion(question: string): Observable<{ response: string }> {
    const body = {
      question,
      session_id: this.sessionId
    };
    return this.http.post<{ response: string }>(this.apiUrl, body);
  }

  private generateSessionId(): string {
    return crypto.randomUUID();
  }
}
