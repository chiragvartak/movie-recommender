import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  userId: string = "";
  constructor(private router: Router) {
    document.body.style.backgroundColor = "#ededed";
  }

  ngOnInit(): void {
  }

  getRecommendationsById() {
    this.router.navigate(['/recommendations', this.userId]);
  }

}
