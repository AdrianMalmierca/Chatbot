import { Component } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  standalone: true,
  imports: [FormsModule, CommonModule],
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent {
  question = '';
  loading = false;
  showSlowWarning = false;
  error = '';

  private slowWarningTimeout: ReturnType<typeof setTimeout> | null = null;

  messages: { sender: 'user' | 'bot'; text: string }[] = [];

  constructor(private chatService: ChatService) {}

  send() {
    const content = this.question.trim();
    if (!content) return;

    this.messages.push({ sender: 'user', text: content });
    this.loading = true;
    this.showSlowWarning = false;
    this.error = '';
    this.question = '';

    //if takes more than 10 seconds, the backend is waking up
    this.slowWarningTimeout = setTimeout(() => {
      this.showSlowWarning = true;
    }, 10000);

    this.chatService.sendQuestion(content).subscribe({
      next: (res) => {
        this.messages.push({ sender: 'bot', text: res.response });
        this.finishLoading();
      },
      error: () => {
        this.error = 'Error al contactar el servidor.';
        this.finishLoading();
      },
    });
  }

  private finishLoading() {
    this.loading = false;
    this.showSlowWarning = false;
    if (this.slowWarningTimeout) {
      clearTimeout(this.slowWarningTimeout);
      this.slowWarningTimeout = null;
    }
  }
}