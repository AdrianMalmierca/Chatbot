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
  error = '';

  messages: { sender: 'user' | 'bot'; text: string }[] = [];

  constructor(private chatService: ChatService) {}

  send() {
    const content = this.question.trim();
    if (!content) return;

    this.messages.push({ sender: 'user', text: content });
    this.loading = true;
    this.error = '';
    this.question = '';

    this.chatService.sendQuestion(content).subscribe({
      next: (res) => {
        this.messages.push({ sender: 'bot', text: res.response });
        this.loading = false;
      },
      error: () => {
        this.error = 'Error al contactar el servidor.';
        this.loading = false;
      },
    });
  }
}
