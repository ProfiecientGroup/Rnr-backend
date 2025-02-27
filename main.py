from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum
from decimal import Decimal
import googlemaps
from itertools import permutations
import re
import math
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Taxi Service API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Constants
BASE_ADDRESS = "46 Eamer Crescent, Wokingham, UK"
CONGESTION_CHARGE = 15.0


class TripType(str, Enum):
    ONE_WAY = "one_way"
    ROUND_TRIP = "round_trip"
    HOURLY = "hourly"


class CarClass(str, Enum):
    E_CLASS = "E_CLASS"
    S_CLASS = "S_CLASS"
    V_CLASS = "V_CLASS"
    ESTATE = "ESTATE"


class Location(BaseModel):
    address: str


class BookingDetails(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: str
    noOfPassenger: str
    noOfSuitcase: str
    message: Optional[str] = None


class BookingRequest(BaseModel):
    pickups: List[Location]
    dropoffs: List[Location]
    trip_type: TripType
    start_datetime: str
    end_datetime: Optional[str] = None
    hours: int = 0
    bookingDetails: BookingDetails


class RouteSegment(BaseModel):
    from_address: str
    to_address: str
    distance_miles: float
    price: float


class CarPrice(BaseModel):
    car_class: str
    total_distance_miles: float
    base_price: float
    car_class_price: float
    congestion_charge: float
    final_price: float
    route_segments: List[RouteSegment]
    calculation_breakdown: str


class PricingResponse(BaseModel):
    prices: List[CarPrice]
    total_distance_miles: float
    route_summary: str
    booking_reference: str
    trip_type: TripType
    hours: int = 0


class TaxiPricingCalculator:
    def __init__(self, google_maps_api_key: str):
        self.gmaps = googlemaps.Client(key=google_maps_api_key)

        # Special locations mapping
        self.special_locations = {
            "heathrow": ["Heathrow Airport", "LHR", "London Heathrow"],
            "gatwick": ["Gatwick Airport", "LGW", "London Gatwick"],
            "luton": ["Luton Airport", "London Luton Airport"],
            "stansted": ["Stansted Airport", "London Stansted"],
            "city airport": ["London City Airport"],
            "st pancras": ["St Pancras International"],
            "farnborough": ["Farnborough Airport"],
            "raf northolt": ["RAF Northolt"],
            "southend": ["Southend Airport"],
            "reading station": ["Reading Railway Station"],
            "twyford station": ["Twyford Railway Station"],
            "bracknell station": ["Bracknell Railway Station"],
            "southampton docks": ["Southampton Docks"]
        }
        # Congestion zone coordinates (approximate central London)
        self.congestion_zone = {
            "sw": {"lat": 51.4898, "lng": -0.1444},  # Southwest corner
            "ne": {"lat": 51.5259, "lng": -0.0857}  # Northeast corner
        }

    def _get_full_address(self, partial_address: str) -> str:
        """Convert partial address to full address using Google Maps"""
        try:
            # First check if it's a special location
            lower_address = partial_address.lower()
            for key, variants in self.special_locations.items():
                if any(variant.lower() in lower_address for variant in variants):
                    # Get the first result from Google Maps
                    result = self.gmaps.geocode(variants[0])
                    if result:
                        return result[0]['formatted_address']

            # If not a special location, use regular geocoding
            result = self.gmaps.geocode(partial_address)
            if result:
                return result[0]['formatted_address']
            return partial_address
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error getting full address: {str(e)}")

    def _calculate_distance(self, origin: str, destination: str) -> float:
        """Calculate distance in miles between two addresses"""
        try:
            result = self.gmaps.distance_matrix(origin, destination, units="imperial")
            if result['rows'][0]['elements'][0]['status'] != "OK":
                raise HTTPException(status_code=400, detail="Unable to calculate distance")
            return result['rows'][0]['elements'][0]['distance']['value'] / 1609.34
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error calculating distance: {str(e)}")

    def _is_in_congestion_zone(self, address: str) -> bool:
        """Check if address falls within congestion zone"""
        try:
            result = self.gmaps.geocode(address)
            if result:
                lat = result[0]['geometry']['location']['lat']
                lng = result[0]['geometry']['location']['lng']
                return (self.congestion_zone['sw']['lat'] <= lat <= self.congestion_zone['ne']['lat'] and
                        self.congestion_zone['sw']['lng'] <= lng <= self.congestion_zone['ne']['lng'])
            return False
        except:
            return False

    def _calculate_callout_or_dropout_charge(self, distance: float) -> Tuple[float, str]:
        """Calculate callout or dropout charge based on distance"""
        explanation = ""

        if distance <= 3:
            charge = 0
            explanation = f"Distance {distance:.1f} miles (≤ 3 miles): £0.00"
        elif distance <= 5:
            charge = 15
            explanation = f"Distance {distance:.1f} miles (3-5 miles): £10.00"
        elif distance <= 10:
            charge = 30
            explanation = f"Distance {distance:.1f} miles (5-10 miles): £15.00"
        else:
            charge_bracket = math.ceil((distance - 10) / 5)
            charge = 10 + (charge_bracket * 1)
            lower_bound = 10 + (charge_bracket - 1) * 5
            upper_bound = lower_bound + 5
            explanation = f"Distance {distance:.1f} miles ({lower_bound}-{upper_bound} miles): £{charge:.2f}"

        return charge, explanation

    def _calculate_journey_price(self, distance: float) -> Tuple[float, str]:
        """Calculate journey price based on distance"""
        explanation = ""

        if distance <= 36:
            rate = 2.5
            explanation = f"Journey distance {distance:.1f} miles (≤ 50 miles): {distance:.1f} × £2.50 = £{distance * rate:.2f}"
        elif distance <= 49:
            rate = 2.9
            explanation = f"Journey distance {distance:.1f} miles (≤ 50 miles): {distance:.1f} × £2.9 = £{distance * rate:.2f}"
        elif distance <= 69:
            rate = 2.0
            explanation = f"Journey distance {distance:.1f} miles (50-100 miles): {distance:.1f} × £2.0 = £{distance * rate:.2f}"
        elif distance <= 89:
            rate = 1.7
            explanation = f"Journey distance {distance:.1f} miles (50-100 miles): {distance:.1f} × £1.7 = £{distance * rate:.2f}"
        else:
            rate = 1.9
            explanation = f"Journey distance {distance:.1f} miles (> 100 miles): {distance:.1f} × £2.00 = £{distance * rate:.2f}"

        return distance * rate, explanation

    def _find_optimal_pickup_route(self, pickup_addresses: List[str]) -> Tuple[List[str], float]:
        """Find the optimal route through all pickup points starting from base address"""
        if not pickup_addresses:
            return [], 0

        best_route = None
        min_distance = float('inf')

        for route in permutations(pickup_addresses):
            total_distance = 0
            current = BASE_ADDRESS

            for next_stop in route:
                distance = self._calculate_distance(current, next_stop)
                total_distance += distance
                current = next_stop

            if total_distance < min_distance:
                min_distance = total_distance
                best_route = list(route)

        return best_route, min_distance

    def _calculate_car_class_price(self, base_price: float, car_class: CarClass) -> tuple[float, str]:
        """Calculate price for specific car class"""
        if car_class == CarClass.E_CLASS:
            return base_price, "E_CLASS base price"
        elif car_class == CarClass.S_CLASS:
            increase = base_price * 0.25
            price = base_price + increase
            return price, f"E_CLASS base (£{base_price:.2f}) + 35% (£{increase:.2f})"
        elif car_class == CarClass.V_CLASS:
            increase = base_price * 0.35
            price = base_price + increase
            return price, f"E_CLASS base (£{base_price:.2f}) + 35% (£{increase:.2f})"
        elif car_class == CarClass.ESTATE:
            increase = base_price * 0.20
            price = base_price + increase
            return price, f"E_CLASS base (£{base_price:.2f}) + 20% (£{increase:.2f})"
        return base_price, "Unknown car class"

    def calculate_prices(self, request: BookingRequest) -> PricingResponse:
        """Calculate prices based on request type"""
        if request.trip_type == TripType.HOURLY:
            raise HTTPException(status_code=400, detail="Hourly bookings not supported with new pricing model")

        # Convert all addresses to full addresses
        pickup_addresses = [self._get_full_address(p.address) for p in request.pickups]
        dropoff_addresses = [self._get_full_address(d.address) for d in request.dropoffs]

        # Find optimal pickup route if there are multiple pickups
        optimal_pickup_route, _ = self._find_optimal_pickup_route(pickup_addresses)

        # Get the callout charge (distance from base to first pickup)
        first_pickup = optimal_pickup_route[0] if optimal_pickup_route else pickup_addresses[0]
        callout_distance = self._calculate_distance(BASE_ADDRESS, first_pickup)
        callout_charge, callout_explanation = self._calculate_callout_or_dropout_charge(callout_distance)

        # Get the journey distance (from last pickup to last dropoff)
        last_pickup = optimal_pickup_route[-1] if optimal_pickup_route else pickup_addresses[0]
        journey_distance = 0
        journey_segments = []

        # Calculate distance between last pickup and all dropoffs
        current = last_pickup
        for dropoff in dropoff_addresses:
            segment_distance = self._calculate_distance(current, dropoff)
            journey_segments.append(RouteSegment(
                from_address=current,
                to_address=dropoff,
                distance_miles=segment_distance,
                price=segment_distance * 3.0  # Use standard rate for showing segment price
            ))
            journey_distance += segment_distance
            current = dropoff

        # Calculate journey price
        journey_price, journey_explanation = self._calculate_journey_price(journey_distance)

        # Get the dropout charge (distance from last dropoff to base)
        last_dropoff = dropoff_addresses[-1]
        dropout_distance = self._calculate_distance(last_dropoff, BASE_ADDRESS)
        dropout_charge, dropout_explanation = self._calculate_callout_or_dropout_charge(dropout_distance)

        # Calculate total distance
        total_distance = callout_distance + journey_distance + dropout_distance

        # Check if any point is in congestion zone
        in_congestion_zone = any(self._is_in_congestion_zone(addr) for addr in pickup_addresses + dropoff_addresses)

        # For round trips
        if request.trip_type == TripType.ROUND_TRIP:
            # For a round trip, double the journey price
            journey_price *= 2
            journey_explanation += f"\nRound trip: ×2 = £{journey_price:.2f}"

            # Add the return journey segments (reversed)
            return_segments = []
            current = last_dropoff
            for dropoff in reversed(dropoff_addresses[:-1]):
                segment_distance = self._calculate_distance(current, dropoff)
                return_segments.append(RouteSegment(
                    from_address=current,
                    to_address=dropoff,
                    distance_miles=segment_distance,
                    price=segment_distance * 3.0
                ))
                journey_distance += segment_distance
                current = dropoff

            for pickup in reversed(optimal_pickup_route):
                segment_distance = self._calculate_distance(current, pickup)
                return_segments.append(RouteSegment(
                    from_address=current,
                    to_address=pickup,
                    distance_miles=segment_distance,
                    price=segment_distance * 3.0
                ))
                journey_distance += segment_distance
                current = pickup

            journey_segments.extend(return_segments)

            # For round trips, we don't add dropout charge since we return to the pickup point
            dropout_charge = 0
            dropout_explanation = "Round trip: No dropout charge (returning to pickup point)"

        # Calculate base price
        base_price = journey_price + callout_charge + dropout_charge

        # Create all route segments including from base to first pickup
        all_segments = [RouteSegment(
            from_address=BASE_ADDRESS,
            to_address=first_pickup,
            distance_miles=callout_distance,
            price=callout_charge
        )]

        # Add segments between pickups if there are multiple
        current = first_pickup
        for pickup in optimal_pickup_route[1:]:
            pickup_segment_distance = self._calculate_distance(current, pickup)
            all_segments.append(RouteSegment(
                from_address=current,
                to_address=pickup,
                distance_miles=pickup_segment_distance,
                price=pickup_segment_distance * 3.0
            ))
            current = pickup

        # Add journey segments
        all_segments.extend(journey_segments)

        # Add the dropout segment if not a round trip
        if request.trip_type != TripType.ROUND_TRIP:
            all_segments.append(RouteSegment(
                from_address=last_dropoff,
                to_address=BASE_ADDRESS,
                distance_miles=dropout_distance,
                price=dropout_charge
            ))

        # Calculate prices for each car class
        prices = []
        for car_class in CarClass:
            car_class_price, car_explanation = self._calculate_car_class_price(base_price, car_class)

            # Add congestion charge if applicable
            congestion_charge = CONGESTION_CHARGE if in_congestion_zone else 0
            final_price = car_class_price + congestion_charge

            # Create price breakdown explanation
            breakdown = (
                f"Journey type: {request.trip_type.value}\n"
                f"Call-out distance: {callout_distance:.1f} miles\n"
                f"Journey distance: {journey_distance:.1f} miles\n"
                f"Drop-out distance: {dropout_distance:.1f} miles\n\n"
                f"1. Call-out charge: {callout_explanation}\n"
                f"2. Journey price: {journey_explanation}\n"
                f"3. Drop-out charge: {dropout_explanation}\n\n"
                f"Base price: £{base_price:.2f}\n"
                f"Car class calculation: {car_explanation}\n"
                f"{'Congestion charge: £15.00' if in_congestion_zone else 'No congestion charge'}\n"
                f"Final price: £{final_price:.2f}"
            )

            prices.append(CarPrice(
                car_class=car_class.value,
                total_distance_miles=total_distance,
                base_price=base_price,
                car_class_price=car_class_price,
                congestion_charge=congestion_charge,
                final_price=final_price,
                route_segments=all_segments,
                calculation_breakdown=breakdown
            ))

        # Create route summary
        route_addresses = [segment.from_address for segment in all_segments]
        route_addresses.append(all_segments[-1].to_address)
        route_summary = "→".join(route_addresses)

        return PricingResponse(
            prices=prices,
            total_distance_miles=total_distance,
            route_summary=route_summary,
            booking_reference=f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}",
            trip_type=request.trip_type,
            hours=request.hours
        )


# Initialize calculator
calculator = TaxiPricingCalculator("AIzaSyAh3BlCUTtjDCrtl9b0cViB6YVX9qKwUxs")


@app.post("/calculate-booking-prices", response_model=PricingResponse)
async def calculate_booking_prices(request: BookingRequest) -> PricingResponse:
    if not request.pickups or not request.dropoffs:
        raise HTTPException(status_code=400, detail="At least one pickup and one dropoff location required")

    return calculator.calculate_prices(request)