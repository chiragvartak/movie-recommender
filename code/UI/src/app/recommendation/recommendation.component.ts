import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-recommendation',
  templateUrl: './recommendation.component.html',
  styleUrls: ['./recommendation.component.css']
})

export class RecommendationComponent implements OnInit {
  movieList: Movie[] = [];
  baseUrl = "http://localhost:5000/" // BASE URL for the backend
  ncfPart = "ncf"; // NCF End point
  hybridPart = "hybrid"; // Hybrid end point
  ncfMovies: Movie; // NCF movies
  hybridMovies: Movie; // HYBRID movies
  userId = null; // User Id

  constructor(private route: ActivatedRoute, private httpClient: HttpClient, private router: Router) {
    document.body.style.backgroundColor = "#000000";
    this.userId = this.route.snapshot.paramMap.get("id");
    let params = new HttpParams().set('userId', this.userId);
    this.httpClient.get<Movie>(this.baseUrl + this.ncfPart, { params: params }).subscribe(ncfData => {
      this.ncfMovies = ncfData;
    });
    this.httpClient.get<Movie>(this.baseUrl + this.hybridPart, { params: params }).subscribe(hybridData => {
      this.hybridMovies = hybridData;
      this.displayContent(0, 1);
    });
  }

  ngOnInit(): void {
    document.getElementById("result-container").style.display = "none";
  }

  /*
  Go to Login screen
  */
  goBack() {
    this.router.navigateByUrl('/login');
  }

  /*
  Scroll Right in the movie list
  */
  scrollRight(elementType: number) {
    let scrollElement = this.getScrollElement(elementType);
    scrollElement.scrollLeft += 300;
  }

  /*
  Scroll left in the movie list
  */
  scrollLeft(elementType: number) {
    let scrollElement = this.getScrollElement(elementType);
    scrollElement.scrollLeft -= 300;
  }

  /*
  Set the scroll left button enabled/disabled
  */
  checkDisabledLeft(elementType: number) {
    let scrollElement = this.getScrollElement(elementType);
    if (scrollElement.scrollLeft < 1) {
      return true;
    }
    return false;
  }

  /*
  Set the scroll right button enabled/disabled
  */
  checkDisabledRight(elementType: number) {
    let scrollElement = this.getScrollElement(elementType);
    if (scrollElement.scrollLeft > 1670) {
      return true;
    }
    return false;
  }

  /*
  Get the selected list container(Hybrid or NCF)
  */
  getScrollElement(elementType: number) {
    let scrollElement = null;
    if (elementType == 1) {
      scrollElement = document.getElementById("hybrid-container");
    } else {
      scrollElement = document.getElementById("nsf-container");
    }
    return scrollElement;
  }

  /*
  Display content of selected movie 
  */
  displayContent(index: number, elementType: number) {
    let movie: Movie = elementType == 1 ? this.hybridMovies : this.ncfMovies;
    if (movie) {
      const src = this.getImgSrc(movie.movie_names[index]);
      document.getElementById("result-container").style.display = "block";
      document.getElementById("current-movie-name").innerHTML = movie.movie_names[index];
      document.getElementById("current-movie-score").innerHTML = movie.scores[index];
      document.getElementById("result-container").style.backgroundImage = "url(" + src + ")";
    }
  }
  /*
  Get image for selected movie
  */
  getImgSrc(movieId: any) {
    switch (movieId) {
      case "Shawshank Redemption, The (1994)":
        return "../../assets/shawshank-redemption.jpg";

      case "Planet Earth (2006)":
        return "../../assets/planet-earth.jpg";

      case "Apartment, The (1960)":
        return "../../assets/the-apartment.jpg";

      case "Schindler's List (1993)":
        return "../../assets/schindlers-list.jpg";

      case "High and Low (Tengoku to jigoku) (1963)":
        return "../../assets/high-low.jpg";

      case "Spotlight (2015)":
        return "../../assets/spotlight.jpg";

      case "Human Condition I, The (Ningen no joken I) (1959)":
        return "../../assets/human-condition.jpg";

      case "Lives of Others, The (Das leben der Anderen) (2006)":
        return "../../assets/the-lives-of-others.jpg";

      case "Iron Giant, The (1999)":
        return "../../assets/iron-giant2.png";

      case "City Hall (1996)":
        return "../../assets/379469.webp";

      case "Hurt Locker, The (2008)":
        return "../../assets/hurtlocker.jpg";

      case "Ruthless People (1986)":
        return "../../assets/ruthless-people.jpg";

      case "Empire Records (1995)":
        return "../../assets/empirerecords1.jpg";

      case "Forces of Nature (1999)":
        return "../../assets/forces.jpg";

      case "Planet Terror (2007)":
        return "../../assets/plterror.jpg";

      case "Planet Earth II (2016)":
        return "../../assets/planet-earth-ii.jpg";

      case "Fight Club (1999)":
        return "../../assets/cult-fight-club.jpg";

      case "Cosmos":
        return "../../assets/782685767.webp";

      case "Heiress, The (1949)":
        return "../../assets/160504_heiress_banner1.jpg";

      case "HyperNormalisation (2016)":
        return "../../assets/HyperNormalisation-871297177-large.jpg";

      case "42 Up (1998)":
        return "../../assets/MV5BNGVlMGE3MmQtNDhjYS00M2UwLWE0YzktOTk2OGQyZjhiZWNjXkEyXkFqcGdeQXVyNjczMzgwMDg@._V1_.jpg";

      case "Scenes From a Marriage (Scener ur ett äktenskap) (1973)":
        return "../../assets/scenesfromamarriage3.jpg";

      case "Harry Potter and the Prisoner of Azkaban (2004)":
        return "../../assets/2086552_1280x800.jpg";

      case "Book of Eli, The (2010)":
        return "../../assets/the-book-of-eli_1600x1200_76249.jpg";

      case "Live Free or Die Hard (2007)":
        return "../../assets/4914d79c06be1_60130b.jpg";

      case "American Pie (1999)":
        return "../../assets/images.jfif";

      case "Time Traveler's Wife, The (2009)":
        return "../../assets/2009196_1280x800.jpg";

      case "Bulletproof Monk (2003)":
        return "../../assets/A6D7B9D8-71F3-431F-AEF4-EF288FC57F66_1024x1024.jpeg";

      case "Long Goodbye, The (1973)":
        return "../../assets/longgoodbye.jpg";

      case "Animal Kingdom (2010)":
        return "../../assets/DAF421D5-DA4B-44E6-B072-582E19EFF8AA.jpeg";

      default:
        return "../../assets/sample.jpg";
    }
  }

}

export class Movie {
  user_id: string;
  latest_interacted_movie_id: string;
  latest_interacted_movie_title: string;
  scores: string[];
  movie_ids: string[];
  movie_names: string[];
  hit_ratio: string;
}